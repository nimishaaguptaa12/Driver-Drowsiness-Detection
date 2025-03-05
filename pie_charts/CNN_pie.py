import matplotlib.pyplot as plt

# Data for CNN Model
labels = ['Yawning', 'Open Eyes', 'Closed Eyes', 'Non-Yawning']
sizes = [85.6, 92.4, 90.2, 94.1]

# Create the pie chart
plt.figure(figsize=(8, 6))  # Adjust size as needed
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('CNN Model Accuracy Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
