from experiments.qual_analysis import *
from experiments.quant_analysis import *
from experiments.downstream import *



def run_experiments(store_res=True):
  # Qualitative analysis of stylized facts
  qualitative_analysis(store_res=store_res)

  # Quantitative analysis of distribution tests
  quantitative_analysis()

  # Downstream experiments of TMTR and TATR
  downstream_experiments(ahead=1, store_fig=store_res)









