library(tidyverse)

dat = read.csv("Z:/fields/Misc/FTW/comparison_result_UnMasked_FTW.csv")
 

dat %>%
  select(-MaskVersion) %>% 

  pivot_longer(cols = c(Mean.Thuenen.polygons, Median.Thuenen.polygons,
                        Mean.Our.polygons, Median.Our.polygons,
                        Mean.FTW.polygons, Median.FTW.polygons),
               names_to = 'key',
               values_to = 'value') %>% 
  separate(key, into = c("measure", "polygon", "dropcol"), sep = "\\.") %>%
  select(-dropcol) %>% 
  mutate(polygon = fct_relevel(polygon, "Our", "Thuenen", "FTW"),
         polygon = fct_recode(polygon, "PTAViT3D" = "Our",
                                  "FracTAL ResUNet" = "Thuenen",
                              "FTW" = "FTW")) %>%
  mutate(measure = fct_relevel(measure, "Mean", "Median"),
         measure = fct_recode(measure, "Mean of IoUs" = "Mean",
                              "Median of IoUs" = "Median")) %>%
  ggplot(aes(x = Threshold.on.overlap.with.Other, y = Threshold.on.overlap.with.IACS, fill = value)) + 
  geom_tile() +
  geom_text(aes(label = round(value, 3)), color = 'red', size = 3) +
  facet_grid(polygon ~ measure) +
  scale_fill_viridis_c() + 
  theme_light() +
  labs(x="min. proportion of polygons' intersection to non IACS polygon",
       y="min. proportion of polygons' intersection to IACS polygon") +
  theme(
    # aspect.ratio = 1,  # Keep square aspect ratio for the facets
    strip.text = element_text(size = 14, color = "black"),
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 14),
    axis.ticks = element_line(linewidth = 1.5),
    plot.title = element_markdown(size = 14, lineheight = 1.5, halign = 0.5))



# 
# 
# ggsave(paste0(storPath, paste0('grid_search_', mod_name, '_', vari, '.png')),
#        plot = p,
#        dpi = 300,
#        width = 40,
#        height = 22,
#        units = "cm",
#        bg = 'grey')