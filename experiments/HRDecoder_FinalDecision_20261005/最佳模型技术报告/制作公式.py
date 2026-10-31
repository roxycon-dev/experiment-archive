import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
equations=json.loads((root/'公式源.json').read_text('utf8'))
for key,latex in equations.items():
    fig=plt.figure(figsize=(12,1.2),dpi=300)
    fig.text(.01,.5,'$'+latex+'$',fontsize=17,va='center',ha='left',color='black')
    fig.savefig(root/'图示'/('公式_'+key+'.png'),dpi=300,bbox_inches='tight',pad_inches=.07,transparent=False)
    fig.savefig(root/'图示'/('公式_'+key+'.svg'),bbox_inches='tight',pad_inches=.07)
    plt.close(fig)
print('Rendered',len(equations),'equations')
