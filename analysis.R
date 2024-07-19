library(tidyr)
library(dplyr)
library(ggplot2)
library(RColorBrewer)

setwd('~/projects/topo-eval/')

### AVERAGE ACROSS LAYERS
df <- read.csv('data/topobert/fedorenko-by-layer-unmasked.csv')

summarized <- df %>%
  group_by(condition) %>%
  summarize(
    mean_activation = mean(activation),
    ci_low = mean_activation - qt(0.95, df=n()-1) * sd(activation) / sqrt(n()),
    ci_high = mean_activation + qt(0.95, df=n()-1) * sd(activation) / sqrt(n())
  )

summarized %>% ggplot(aes(x = condition, y = mean_activation, fill = condition)) +
  geom_bar(stat = 'identity') +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.2, color = 'black') +
  scale_fill_brewer(palette = 'Greens') +
  theme_bw()

ggsave('figures/topobert/fedorenko-response-unmasked.pdf')

### INCLUDE LAYERS
summarized <- df %>%
  group_by(condition, layer) %>%
  summarize(
    mean_activation = mean(activation),
    ci_low = mean_activation - qt(0.95, df=n()-1) * sd(activation) / sqrt(n()),
    ci_high = mean_activation + qt(0.95, df=n()-1) * sd(activation) / sqrt(n())
  )

summarized %>% ggplot(aes(x = condition, y = mean_activation, fill = condition)) +
  facet_wrap(~layer) +
  geom_bar(stat = 'identity') +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.2, color = 'black') +
  scale_fill_brewer(palette = 'Greens') +
  theme_bw()

ggsave('figures/topobert/fedorenko-response-by-layer.pdf')

### BRAIN

df <- read.csv('data/topobert/shain-results.csv')
summarized <- df %>%
  group_by(Effect) %>%
  summarize(
    mean_activation = mean(EffectSize),
    ci_low = mean_activation - qt(0.95, df=n()-1) * sd(EffectSize) / sqrt(n()),
    ci_high = mean_activation + qt(0.95, df=n()-1) * sd(EffectSize) / sqrt(n())
  )

summarized %>% ggplot(aes(x = Effect, y = mean_activation, fill = Effect)) +
  geom_bar(stat = 'identity') +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.2, color = 'black') +
  scale_fill_brewer(palette = 'Greens', name = 'condition') +
  ylab('bold_signal_change') +
  xlab('condition') +
  theme_bw()

ggsave('figures/topobert/fedorenko-response-brain.png')