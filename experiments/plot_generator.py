import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
sns.set_palette("colorblind")

def create_professional_plots(json_file):
    results_dir = Path(json_file).parent
    
    with open(json_file, 'r') as f:
        all_results = json.load(f)
        
    scenarios = [s for s in all_results.keys() if s not in ['overall_comparison', 'meta']]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('TAVS vs. Traditional Full Verification (10-Round FL, CIFAR-10)', fontsize=18, fontweight='bold')
    
    # 1. Verification Overhead (Time)
    ax = axes[0]
    tavs_times = []
    full_times = []
    
    for s in scenarios:
        # Extract time string from VerificationResults(...) repr in the JSON
        tavs_str = all_results[s]['tavs']
        full_str = all_results[import json
import matplotlib.pyplot as plt
import seaborn as sns
import nump  import mate import seaborn as sns
import nveimport numpy as np
f].from pathlib impo  
# Set style
plt.style.ullplt.style.("sns.set_context("paper", font_scale=1.")[0])
        
        tavs_times.append(t
def create_professional_plo.ap    results_dir = Path(json_file).parentge    
    with open(json_file, 'r') as f     x.        all_results = json.load(f),         
    scenarios = [s for sge    scebl    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('TAVS vs. Tradiff   e'    fig.suptitle('TAVS vs. Traditional Full Verifiab    
    # 1. Verification Overhead (Time)
    ax = axes[0]
    tavs_times = []
    full_times = []
    
    for )
    a    ax = axes[0]
    tavs_times = []ls    tavs_times , ' ').title() for s i    
    for s in ax   ge        # Extract timebe        tavs_str = all_results[s]['tavs']
        full_str = all_results[iments = 8  # Extracted from our config
    fuimport matplotlib.pyplot as plt(['TAVS', 'Timport seaborn as sns
import nulimport nump  import #2import nveimport numpy as np
f].from pathlib 0.f].from pathlib impo  
# Seab# Set style
plt.stylepeplt.style.fo        
        tavs_times.append(t
def create_professional_plo.ap ='         def create_professional_pl    # Annotate savings
    savings = (full_clients - tavs_clients) / full_clien    scenarios = [s for sge    scebl    
    fig, axes = plt.subplots(1, 3, figsize=( x    fig, axes = plt.subplots(1, 3, fig=(    fig.suptitle('TAVS vs. Tradiff   e'    fig.suppr    # 1. Verification Overhead (Time)
    ax = axes[0]
    tavs_times = []
    full_times = []',    ax = axes[0]
    tavs_times = []d'    tavs_times ol    full_times = [Fi    
    for )
   
    ax    a   2]    tavs_times = []l      for s in ax   ge        # Extract timebe        tavs_str =         full_str = all_results[iments = 8  # Extracted from our coification']
            fuimport matplotlib.pyplot as plt(['TAVS', 'Timport seaborn as snloimport nulimport nump  import #2import nveimport numpy as np
f].from = f].from pathlib 0.f].from pathlib impo  
# Seab# Set style
ep# Seab# Set style
plt.stylepeplt.style.:
plt.stylepeplt.sc         tavs_times.append(t
deacdef crea)[1].split(",")[0])
    savings = (full_clients - tavs_clients) / full_clien    scenarios = [s for sge    scesp    fig, axes = plt.subplots(1, 3, figsize=( x    fig, axes = plt.subplots(1, 3, fig=(    fig.li    ax = axes[0]
    tavs_times = []
    full_times = []',    ax = axes[0]
    tavs_times = []d'    tavs_times ol    full_times = [Fi    
    for )
   
    ax    a   2]    ta,     tavs_times ol    full_times = [co    tavs_times = []d'    tavs_times  +    for )
   
    ax    a   2]    tavs_times = []l      for se', edgecolo  'b            fuimport matplotlib.pyplot as plt(['TAVS', 'Timport seaborn as snloimport nulimport nump  import #2import nveimport numpy as np
old')
    ax.set_xticks(x)
    ax.set_xf].from = f].from pathlib 0.f].from pathlib impo  
# Seab# Set style
ep# Seab# Set style
plt.stylepeplt.style.:
plt.stylepeplt.sc         te# Seab# Set style
ep# Seab# Set style
plt.stylepefoep# Seab# Set stioplt.stylepeplt.styinplt.stylepeplt.sc    
 deacdef crea)[1].split(",")[0])
    savings %"    savings = (full_clients - ta    tavs_times = []
    full_times = []',    ax = axes[0]
    tavs_times = []d'    tavs_times ol    full_times = [Fi    
    for )
   
    ax    a   2]    ta,     tavs_times ol    full_times = [co    ta='    full_times = [f"    tavs_times = []d'    tavs_times }"    for )
   
    ax    a   2]    ta,     tavs_times ol    fu/U   
    na  /g   
    ax    a   2]    tavs_times = []l      for se', edgecolo  'b            fuimport matplon')
