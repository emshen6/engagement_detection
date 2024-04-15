import cv2
import numpy as np
import tensorflow as tf
from scipy.special import softmax

model = tf.keras.models.load_model('Xception_on_DAiSEE_finetune_fc.h5')

class_labels = {0: "Very Low", 1: "Low", 2:"High", 3: "Very High"}

cap = cv2.VideoCapture(0)

cv2.namedWindow('Engagement Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Engagement Detection', 1000, 800)


while True:
    ret, frame = cap.read()

    image = cv2.resize(frame, (299, 299))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    image = tf.keras.applications.xception.preprocess_input(image)

    predictions = model.predict(image)
    print(predictions)

    probabilities = softmax(predictions, axis=-1)

    boredom_predictions = probabilities[0]
    engagement_predictions = probabilities[1]
    confusion_predictions = probabilities[2]
    frustration_predictions = probabilities[3]

    text_lines = [
    f'Boredom Level: {class_labels[np.argmax(boredom_predictions)]}, Probability: {round(np.max(boredom_predictions)*100)}%',
    f'Engagement Level: {class_labels[np.argmax(engagement_predictions)]}, Probability: {round(np.max(engagement_predictions)*100)}%',
    f'Confusion Level: {class_labels[np.argmax(confusion_predictions)]}, Probability: {round(np.max(confusion_predictions)*100)}%',
    f'Frustration Level: {class_labels[np.argmax(frustration_predictions)]}, Probability: {round(np.max(frustration_predictions)*100)}%',
    ]

    y_coordinate = 30

    for line in text_lines:
        cv2.putText(frame, line, (10, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_coordinate += 30

    cv2.imshow('Engagement Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
