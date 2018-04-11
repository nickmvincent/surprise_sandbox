import matplotlib.pyplot as plt
import json

with open('ml-1m_user_to_num_ratings.json', 'r') as f:
    d = json.load(f)
plt.hist(list(d.values()), 100, color='g')
plt.show()