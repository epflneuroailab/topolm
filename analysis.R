library(tidyr)
library(dplyr)
library(ggplot2)
library(RColorBrewer)


setwd('~/projects/topo-eval/')
df <- read.csv('responses_out.csv')

summarized <- df %>%
  group_by(condition) %>%
  summarize(
    mean_mean_activation = mean(mean_activation),
    ci_low = mean_mean_activation - qt(0.95, df=n()-1) * sd(mean_activation) / sqrt(n()),
    ci_high = mean_mean_activation + qt(0.95, df=n()-1) * sd(mean_activation) / sqrt(n())
  )

# scuffed need to change
summarized$mean_activation <- summarized$mean_mean_activation

summarized %>% ggplot(aes(x = condition, y = mean_activation, fill = condition)) +
  geom_bar(stat = 'identity') +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.2, color = 'black') +
  scale_fill_brewer(palette = 'Greens') +
  theme_bw()