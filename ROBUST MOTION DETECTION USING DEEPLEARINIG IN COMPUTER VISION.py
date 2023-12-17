import cv2

# create video capture object
cap = cv2.VideoCapture(0)

# initialize variables
frame_count = 0
motion_detected = False
prev_frame = None

while True:
    # read frame from video
    ret, frame = cap.read()
    
    # check if frame is valid
    if not ret:
        break
    
    # convert frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # if previous frame is None, initialize it
    if prev_frame is None:
        prev_frame = gray
        continue
    
    # calculate absolute difference between current and previous frame
    frame_diff = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    
    # dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # find contours of thresholded image
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # loop over contours
    for contour in contours:
        # if contour area is too small, ignore it
        if cv2.contourArea(contour) < 500:
            continue
        
        # motion detected
        motion_detected = True
        
        # draw bounding box around contour
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
    # update previous frame
    prev_frame = gray
    
    # increment frame count
    frame_count += 1
    
    # display the resulting frame
    cv2.imshow('frame', frame)
    
    # check if motion was detected for at least 10 frames
    if motion_detected and frame_count >= 10:
        print("Motion detected!")
        # reset motion detection variables
        frame_count = 0
        motion_detected = False
    
    # check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release video capture object and close windows
cap.release()
cv2.destroyAllWindows()