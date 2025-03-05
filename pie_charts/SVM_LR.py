import matplotlib.pyplot as plt

# Data for SVM Model (Logistic Regression)
labels = ['Yawning', 'Open Eyes', 'Closed Eyes', 'Non-Yawning']
sizes = [82.3, 88.9, 85.7, 91.2]

# Create the pie chart
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('SVM Model (Logistic Regression) Accuracy Distribution')
plt.axis('equal')
plt.show()
