from experiments.tmtr import *
from experiments.tatr import *
from models.model_params import *



##########################
# Downstream experiments #
##########################

def downstream_experiments(ahead=1, store_fig=False):
  """ Conduct the downstream experiments """
  dataname = prm_params['dataname']
  downstream_tmtr(dataname, ahead, store_fig)
  downstream_tatr(dataname, ahead, store_fig)



#########################################
# Train on Mixture, Test on Real (TMTR) #
#########################################

def downstream_tmtr(dataname, ahead=1, store_fig=True):
  """ Downstream experiments under the TMTR setting """
  n_runs = 100
  n_proportions = 10
  mix_length = 252 * 5
  n_epochs = 100 #200
  window_size = 64
  lstm_hidden_dim = 32
  lstm_loss = 'mae'
  TMTR(n_runs, n_proportions, mix_length, n_epochs, window_size, ahead, lstm_hidden_dim, lstm_loss)
  df_res_summary = summarize_results('tmtr', dataname, 'fts-diffusion', ahead, lstm_hidden_dim, lstm_loss)
  plot_dowmstream_tmtr(df_res_summary, store_fig)
  if store_fig:
    print("Results of Downstream TMTR stored.")



##############################################
# Train on Augmentation, Test on Real (TATR) #
##############################################

def downstream_tatr(dataname, ahead=1, store_fig=True):
  """ Downstream experiments under the TATR setting """
  n_runs = 100
  n_augmentations = 100
  aug_length = 252
  n_epochs = 100 #200
  window_size = 64
  lstm_hidden_dim = 32
  lstm_loss = 'mae'
  TATR(n_runs, n_augmentations, aug_length, n_epochs, window_size, ahead, lstm_hidden_dim, lstm_loss)
  df_res_summary = summarize_results('tatr', dataname, 'fts-diffusion', ahead, lstm_hidden_dim, lstm_loss)
  plot_dowmstream_tatr(df_res_summary, store_fig)
  if store_fig:
    print("Results of Downstream TATR stored.")









