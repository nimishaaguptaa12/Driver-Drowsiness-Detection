import matplotlib.pyplot as plt

# Data for SVM Model (Naive Bayes)
labels = ['Yawning', 'Open Eyes', 'Closed Eyes', 'Non-Yawning']
sizes = [80.1, 86.5, 84.4, 89.3]

# Create the pie chart
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('SVM Model (Naive Bayes) Accuracy Distribution')
plt.axis('equal')
plt.show()
