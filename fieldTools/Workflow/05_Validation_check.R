library(tidyverse)

state = "Brandenburg"
model = "AI4_RGB_exclude_True_38"
year = "2023"

folders = list.dirs(paste0('Z:/fields/05_GridSearch/', state, '/', model, '/', year, '/'))
folders <- folders[grepl("result", folders)]

conti = list()

for (folder in folders){
  files = list.files(folder, pattern = '.csv', full.names = T)
  conti[[folder]] = map_dfr(files, read_csv)
}

conti_all <- bind_rows(conti, .id = "source_folder")
rm(conti)


# subset the dataset to comparisons between IACS and predicted polygons, where the area of 
# intersection between those is larger than 50% of the total area of the

# a) predicted polygon
conti_ratio_pred = conti_all %>% 
  filter(reference_field_sizes > 5, ratio_intersect_area_pred > 0.5) %>% 
  group_by(source_folder, t_ext, t_bound) %>% 
  summarise(mean_IoU_max = mean(max_IoU), mean_IoU_centroid = mean(centroid_IoU), sample_size = n())

# b) IACS polygon
conti_ratio_true = conti_all %>% 
  filter(reference_field_sizes > 5, ratio_intersect_area_true > 0.5) %>% 
  group_by(source_folder, t_ext, t_bound) %>% 
  summarise(mean_IoU_max = mean(max_IoU), mean_IoU_centroid = mean(centroid_IoU), sample_size = n())


# rename the source folder to sth shorter
conti_ratio_pred = conti_ratio_pred %>%
  mutate(
    source_folder = sapply(source_folder, function(x) rev(strsplit(x, "/")[[1]])[2])
  )
conti_ratio_true = conti_ratio_true %>%
  mutate(
    source_folder = sapply(source_folder, function(x) rev(strsplit(x, "/")[[1]])[2])
  )

