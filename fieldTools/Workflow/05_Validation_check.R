library(tidyverse)
library(ggtext)


LOCAL_R = FALSE

if(LOCAL_R == TRUE){
  origin = '/home/florus/Aldhani_DATA/'
  home = '/home/florus/Aldhani_HOME/'
}else{
  origin = 'Z:/'
  home = 'Y:/'
}

storPath = paste0(home, 'repos/FieldWaterUseTools/fieldTools/')
state = "Brandenburg"

models = c(
  "AI4_RGB_exclude_True_38",
  'FromScratch_IACS_dilate_False_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_22',
  'FromScratch_IACS_dilate_True_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_47',
  'IACS_dilate_False_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_40_on_AI4_RGB_exclude_True_38_FREEZER_2',
  'IACS_dilate_True_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_40_on_AI4_RGB_exclude_True_38_FREEZER_2',
  'IACS_dilate_False_overlap_40_on_FromScratch_IACS_dilate_True_with_overlap_47_FREEZER_2')

mod_id = c(
  'AI4B_baseline',
  'FromScratch_dilate_F',
  'FromScratch_dilate_T',
  'Finetune_dilate_F',
  'Finetune_dilate_T',
  'Finetune_dilate_F_fromscratch')

year = "2023"

i=1
for (i in seq_along(models)){
 
  mod_name = mod_id[i]
  model = models[i]
  
  folders = list.dirs(paste0(origin, 'fields/05_GridSearch/', state, '/', model, '/', year, '/'))
  folders <- folders[grepl("results2", folders) & grepl("256_20", folders)] #  & !grepl("results2", folders)
  conti = list()
  
  for (folder in folders){

    files = list.files(folder, pattern = '.csv', full.names = T)
    conti[[folder]] = map_dfr(files, read_csv, show_col_types = FALSE)
  }
  
  conti_all <- bind_rows(conti, .id = "source_folder") %>% 
    filter(reference_field_IDs != 0)
  rm(conti)
  

  # subset the dataset to comparisons between IACS and predicted polygons, where the area of 
  # intersection between those is larger than 50% of the total area of the
  
  # a) predicted polygon
  conti_ratio_pred = conti_all %>%
    filter(reference_field_sizes > 5, ratio_intersect_area_pred > 0.5) %>%
    group_by(source_folder, t_ext, t_bound) %>%
    summarise(mean_IoU_max = mean(max_IoU), mean_IoU_centroid = mean(centroid_IoU), sample_size = n()) %>%
    pivot_longer(
      cols = starts_with('IoU') | starts_with('me'),
      names_to = 'metric',
      values_to = 'value'
    )


  # b) IACS polygon
  conti_ratio_true = conti_all %>%
    filter(reference_field_sizes > 5, ratio_intersect_area_true > 0.5) %>%
    group_by(source_folder, t_ext, t_bound) %>%
    summarise(mean_IoU_max = mean(max_IoU), mean_IoU_centroid = mean(centroid_IoU), sample_size = n()) %>%
    pivot_longer(
      cols = starts_with('IoU') | starts_with('me'),
      names_to = 'metric',
      values_to = 'value'
    )
  
  # c) IACS polygon and predicted polygon
  conti_ratio_predtrue = conti_all %>% 
    filter(reference_field_sizes > 5, ratio_intersect_area_true > 0.5, ratio_intersect_area_pred > 0.5) %>% 
    group_by(source_folder, t_ext, t_bound) %>% 
    summarise(mean_IoU_max = mean(max_IoU), mean_IoU_centroid = mean(centroid_IoU), sample_size = n()) %>% 
    pivot_longer(
      cols = starts_with('IoU') | starts_with('me'),
      names_to = 'metric', 
      values_to = 'value'
    )
  
  # d) no restrictions
  conti_ratio_norest = conti_all %>% 
    filter(reference_field_sizes > 10, ratio_intersect_area_true > 0, ratio_intersect_area_pred > 0) %>% 
    group_by(source_folder, t_ext, t_bound) %>%
    summarise(mean_IoU_max = mean(max_IoU), mean_IoU_centroid = mean(centroid_IoU), sample_size = n()) %>%
    pivot_longer(
      cols = starts_with('IoU') | starts_with('me'),
      names_to = 'metric',
      values_to = 'value'
    )


  # rename the source folder to sth shorter
  conti_ratio_pred = conti_ratio_pred %>%
    mutate(
      source_folder = sapply(source_folder, function(x) rev(strsplit(x, "/")[[1]])[2])
    )
  conti_ratio_true = conti_ratio_true %>%
    mutate(
      source_folder = sapply(source_folder, function(x) rev(strsplit(x, "/")[[1]])[2])
    )
  conti_ratio_predtrue = conti_ratio_predtrue %>%
    mutate(
      source_folder = sapply(source_folder, function(x) rev(strsplit(x, "/")[[1]])[2])
    )
  conti_ratio_norest = conti_ratio_norest %>%
    mutate(
      source_folder = sapply(source_folder, function(x) rev(strsplit(x, "/")[[1]])[2])
    )
  
  blocks = list(conti_ratio_pred, conti_ratio_true, conti_ratio_predtrue, conti_ratio_norest)
  modi = c('Predicted', 'IACS', 'Predicted_and_IACS', 'no restriction')
  
  blocks_all = bind_rows(blocks, .id = "orig") %>% 
    mutate(orig = fct_recode(orig,
                            'Ov > 50% Predicted' = "1",
                            'Ov > 50% IACS' = "2",
                            'Ov > 50% Pred & IACS' = "3",
                            'No restrict' = "4"))

  varis = unique(blocks_all$metric)

  for (j in seq_along(varis)){
  
    vari = varis[j]
    
    p <- blocks_all %>%
      filter(source_folder %in% c('ThuenenMask_256_20', 'unmasked_chips_256_20'),
             metric == vari) %>%
      ggplot(aes(x = t_ext, y = t_bound, fill = value)) +
      geom_tile() +
      geom_text(aes(label = round(value*10000, 0)), color = 'red', size = 3) +
      facet_wrap(source_folder ~ orig, ncol = 4) +
      scale_fill_viridis_c() +
      labs(
        title = paste0(
          '**GridSearch for Bandenburg (2023)** prediction based on model
      <br>**', mod_name, '.**',
          '<br>Red values show mean of **',vari,'**. Furthermore, only polygons > 5 pixel were selected.
      <br>-------------------------------------------------------------------------------------------------------------------------------------'),
        x = 't_ext', y = 't_bound'
      ) +
      theme_minimal() +
      theme(
        # aspect.ratio = 1,  # Keep square aspect ratio for the facets
        strip.text = element_text(size = 14, color = "black"),
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 14),
        axis.ticks = element_line(linewidth = 1.5),
        plot.title = element_markdown(size = 14, lineheight = 1.5, halign = 0.5))

    ggsave(paste0(storPath, paste0('grid_search_', mod_name, '_', vari, '.png')),
           plot = p,
           dpi = 300,
           width = 40,
           height = 22,
           units = "cm",
           bg = 'grey')
    
    p <- blocks_all %>%
      filter(source_folder %in% c('ThuenenMask_256_20', 'unmasked_chips_256_20'),
             metric == vari) %>%
      ggplot(aes(x = t_ext, y = t_bound, fill = value)) +
      geom_tile() +
      geom_text(aes(label = sample_size), color = 'black', size = 3) +
      facet_wrap(source_folder ~ orig, ncol = 4) +
      scale_fill_viridis_c() +
      labs(
        title = paste0(
          '**GridSearch for Bandenburg (2023)** prediction based on model
      <br>**', mod_name, '.**',
          '<br>Black values show sample sizes of **',vari,'**. Furthermore, only polygons > 5 pixel were selected.
      <br>-------------------------------------------------------------------------------------------------------------------------------------'),
        x = 't_ext', y = 't_bound'
      ) +
      theme_minimal() +
      theme(
        strip.text = element_text(size = 14, color = "black"),
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 14),
        axis.ticks = element_line(linewidth = 1.5),
        plot.title = element_markdown(size = 14, lineheight = 1.5, halign = 0.5))

    ggsave(paste0(storPath, paste0('grid_search_', mod_name, '_', vari, '_samplesizes.png')),
           plot = p,
           dpi = 300,
           width = 40,
           height = 22,
           units = "cm",
           bg = 'grey')
  }
  print('Next model')
}






############################################################### legacy
for (i in seq_along(blocks)){
  block <- blocks[[i]]
  mod <- modi[i]
  for(vari in c("mean_IoU_max", "mean_IoU_centroid")){
    
    p <- block %>% 
      filter(metric==vari) %>% 
      ggplot(aes(x = t_ext, y = t_bound, fill = value)) + 
      geom_tile() +
      geom_text(aes(label = round(value, 2)), color = 'red', size = 3) +
      facet_wrap(~ source_folder, ncol = 3) +
      scale_fill_viridis_c() + 
      labs(
        title = paste0('**GridSearch for Bandenburg (2023)** prediction based on model **', model, '**',
                       '<br>Only polygons > 5 pixel and where area of intersection between IACS and predicted polygons is larger than
                   <br>50% of the total area of **', mod, '** polygon. Red values show mean of **', vari,'** (min sample size = ', min(block$sample_size), ')'),
        x = 't_ext', y = 't_bound'
      ) + 
      theme_minimal() +
      theme(
        aspect.ratio = 1,  # Keep square aspect ratio for the facets
        strip.text = element_text(size = 14, color = "black"),
        axis.text = element_text(size = 14),                         
        axis.title = element_text(size = 14),         
        axis.ticks = element_line(linewidth = 1.5),
        plot.title = element_markdown(size = 14, lineheight = 1.5))
    
    ggsave(paste0(storPath, paste0('grid_search_', model, '_', vari,'_', mod, '.png')), 
           plot = p, 
           dpi = 900,            
           width = 32,        
           height = 22,          
           units = "cm",        
           bg = 'grey')      
  }
}
