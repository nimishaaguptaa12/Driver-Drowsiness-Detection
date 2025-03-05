import matplotlib.pyplot as plt

# Data for MobileNet Model
labels = ['Yawning', 'Open Eyes', 'Closed Eyes', 'Non-Yawning']
sizes = [88.7, 94.2, 92.8, 96.0]

# Create the pie chart
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('MobileNet Model Accuracy Distribution')
plt.axis('equal')
plt.show()
