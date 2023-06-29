import os
import sys
import cv2


class SelectSquare:
    # The coordinates defining the square selected will be kept in this list.
    select_coords = []
    # While we are in the process of selecting a region, this flag is True.
    selecting = False

    def __init__(self, frame, window_name, size_px=50):
        self.frame = frame
        self.img_height, self.img_width = frame.shape[:2]
        self.window = window_name
        self.roi_size = int(size_px/2)
        # Name the main frame window after the frame filename.
        cv2.namedWindow(self.window, cv2.WINDOW_KEEPRATIO)
        # cv2.resizeWindow(self.window, 1280, 720)
        cv2.setMouseCallback(self.window, self.region_selection)
        # Keep looping and listening for user input until 'c' is pressed.
        cv2.imshow(self.window, self.frame)
        while True:
            key = cv2.waitKey(1) & 0xFF
            # If 'c' is pressed, break from the loop and handle any region selection.
            if key == ord("c"):
                cv2.destroyAllWindows()
                break

    def get_coords(self):
        cx, cy = self.select_coords[0]
        x, y = self.select_coords[1]
        x0, y0, x1, y1 = self.get_square_coords(x, y, cx, cy)
        return [x0, y0, x1-x0, y1-y0]

    def get_square_coords(self, x, y, cx, cy):
        """
        Get the diagonally-opposite coordinates of the square.
        (cx, cy) are the coordinates of the square centre.
        (x, y) is a selected point to which the largest square is to be matched.

        """

        # Selected square edge half-length; don't stray outside the frame boundary.
        a = max(abs(cx-x), abs(cy-y))
        a = min(a, self.img_width-cx, cx, self.img_height-cy, cy)
        return cx-a, cy-a, cx+a, cy+a

    def region_selection(self, event, x, y, flags, param):
        """Callback function to handle mouse events related to region selection."""
        
        # Left mouse button down: begin the selection.
        # The first coordinate pair is the centre of the square.
        # print(self.select_coords)
        if event == cv2.EVENT_LBUTTONDOWN:
            
            self.select_coords = [(x, y)]
            self.selecting = True

        # If we're dragging the selection square, update it.
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
            # pass
            img = self.frame.copy()
            x0, y0, x1, y1 = self.get_square_coords(x, y, *self.select_coords[0])
            if (x1-x0) <= self.roi_size * 2 and (y1-y0) <= self.roi_size * 2:
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                # Display the frame and wait for a keypress
                cv2.imshow(self.window, img)

        # Left mouse button up: the selection has been made.
        elif event == cv2.EVENT_LBUTTONUP:
            x0 = self.select_coords[0][0]
            y0 = self.select_coords[0][1]
            if (x - x0) <= self.roi_size and (y - y0) <= self.roi_size:
                self.select_coords.append((x, y))
            else:
                self.select_coords.append((x0 + self.roi_size, y0 + self.roi_size))
            self.selecting = False


if __name__ == "__main__":
    filename = "../utils/test.jpg"
    basename = os.path.basename(filename)
    image = cv2.imread(filename)
    rect = SelectSquare(image, basename).get_coords()
    print(rect)
