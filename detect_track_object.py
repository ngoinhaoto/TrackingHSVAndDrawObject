import cv2
import numpy as np
import imutils

def callback(value):
    pass

def setup_trackbar(range_filter):
    cv2.namedWindow("Trackbars")

    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 360

        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 360, callback)

def get_trackbar_values(range_filter):
    values = []

    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)

    return values

def main():
    range_filter = ["H", "S", "V"]

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    setup_trackbar(range_filter)
    prev_center = None

    # Create a canvas to draw the lines (initialize it as a black image)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    while True:
        _, frame = cap.read()

        blur = cv2.GaussianBlur(frame, (11, 11), 0)
        frame_to_thresh = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(range_filter)

        thresh = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

            if radius > 10:
                center = (int(x), int(y))

                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)

                # Draw the line on the canvas
                if prev_center is not None:
                    cv2.line(canvas, prev_center, center, (0, 189, 60), 4)

                prev_center = center

        # Combine the canvas and frame to display the lines on the frame
        result = cv2.addWeighted(frame, 1, canvas, 1, 0)
        cv2.imshow("frame", result)
        cv2.imshow("thres", thresh)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if __name__ == "__main__":
    main()
