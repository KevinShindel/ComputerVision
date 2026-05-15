import cv2
import matplotlib.pyplot as plt


def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)


def displayRectangle(frame, bbox):
    plt.figure(figsize=(20, 10))
    frameCopy = frame.copy()
    drawRectangle(frameCopy, bbox)
    frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)
    plt.imshow(frameCopy)
    plt.axis("off")


def drawText(frame, txt, location, color=(50, 170, 50)):
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)


def show_roi_instructions(frame, win_name="Instructions"):
    overlay = frame.copy()

    msg1 = "Please select ROI by mouse."
    msg2 = "Press ENTER/SPACE to confirm or ESC to cancel."

    # Draw a filled rectangle as a text background (top-left)
    cv2.rectangle(overlay, (10, 10), (980, 90), (0, 0, 0), thickness=-1)
    cv2.putText(overlay, msg1, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(overlay, msg2, (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow(win_name, overlay)
    cv2.waitKey(2000)  # show for ~0.7s (adjust or set to 0 to wait for a key)
    cv2.destroyWindow(win_name)
