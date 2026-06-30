import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

data = {
    'order_id': list(range(1, 21)),
    'amount'  : [450,520,480,510,490,530,470,500,2500,515,
                 488,502,478,525,495,505,485,512,50,498],
    'quantity': [2,3,2,3,2,4,2,3,25,3,2,3,2,4,2,3,2,3,1,0]
}
df = pd.DataFrame(data)

high_qty = df[df['quantity']>3]['amount']
lower_qty = df[df['quantity']<=3]['amount']
# print(high_qty,lower_qty)
print(high_qty.mean())
print(lower_qty.mean())

t_test,p_value = stats.ttest_ind(high_qty,lower_qty)
print(t_test)
print(p_value)
if p_value < 0.05:
    print('Reject the Hypothesis')
else:
    print('Fail to Reject the Hypothesis')