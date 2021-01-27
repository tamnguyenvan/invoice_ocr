"""
"""
import os
import cv2
import numpy as np
import imutils
import argparse

tessdata_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tessdata'))
os.environ['TESSDATA_PREFIX'] = tessdata_dir
import pytesseract
import easyocr


class FourPointTransformer:
    """Warp image by four point"""
    def _order_points(self, pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect


    def transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        gap = 10
        dst = np.array([
            [gap, gap],
            [maxWidth - gap, gap],
            [maxWidth - gap, maxHeight - gap],
            [gap, maxHeight - gap]], dtype = "float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        # return the warped image
        return warped

class RectangleDetector:
    def detect(self, c):
        """
        """
        shape = 'undefined'
        peri = cv2.arcLength(c, False)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            return x, y, w, h


class InvoiceDetector:
    def detect(self, img):
        """Detect invoice from given BGR image

        Returns four points of corners
        """
        if not isinstance(img, np.ndarray):
            print('Input image must be a numpy array')
            return

        # Set minimum and max HSV values to display
        lower = np.array([0, 70, 0])
        upper = np.array([179, 255, 255])

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_not(mask)
        N = 3
        mask[:N, :] = 0
        mask[h-N:, :] = 0
        mask[:, :N] = 0
        mask[:, w-N:] = 0

        # Detect contours
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        img_copy = img.copy()
        cv2.drawContours(img_copy, cnts, -1, (0, 255, 0), 1)

        rect_detector = RectangleDetector()
        rect_cnts = []
        for c in cnts:
            result = rect_detector.detect(c)
            if result is not None:
                rect_cnts.append(c)
        
        # Find biggest rectangle
        invoice_cnt = sorted(rect_cnts, key=cv2.contourArea, reverse=True)[0]

        # If possible, get four corners
        peri = cv2.arcLength(invoice_cnt, True)
        approx = cv2.approxPolyDP(invoice_cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
        else:
            x, y, w, h = cv2.boundingRect(invoice_cnt)
            pts = np.array([
                [x, y], [x, y+h],
                [x + w, y + h], [x + w, y]
            ], dtype='int32')

        transformer = FourPointTransformer()
        warped_img = transformer.transform(img, pts)
        return warped_img, pts[0].tolist()


class InvoiceOCR:
    def __init__(self, method='tesseract'):
        self.method = method
        self.reader = None
        if self.method == 'easyocr':
            self.reader = easyocr.Reader(['en'])
    
    def recognize(self, img):
        """
        """
        results = []
        if self.method == 'tesseract':
            data = pytesseract.image_to_data(img, lang='eng',
                                             output_type=pytesseract.Output.DICT)

            num_texts = len(data['level'])
            for i in range(num_texts):
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                text = data['text'][i]
                if text is not None and isinstance(text, str) and len(text) > 0:
                    results.append(((x, y, w, h), text))
        elif self.method == 'easyocr':
            data = self.reader.readtext(img)
            for rs in data:
                cnt = rs[0]
                text = rs[1]
                x, y = cnt[0]
                br = cnt[-1]
                w, h = br[0] - x, br[1] - h
                if text is not None and isinstance(text, str) and len(text) > 0:
                    results.append((x, y, w, h), text)
        else:
            raise ValueError(f'Unsupported method: {self.method}')
        return results


def draw_texts(img, data):
    """
    """
    for box_and_text in data:
        box, text = box_and_text
        box = list(map(int, box))
        box = [max(x, 0) for x in box]
        x, y, w, h = box
        cv2.putText(img, text, (x + w // 3, y + h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def show_img(img):
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Path to image')
    parser.add_argument('-m', '--method', type=str, choices=['tesseract', 'easyocr'],
                        default='tesseract',
                        help='Recognition method')
    args = parser.parse_args()

    # Load image
    img_path = args.input
    img = cv2.imread(img_path)
    
    # Detect invoice contour
    invoice_detector = InvoiceDetector()
    invoice_roi = invoice_detector.detect(img)
    if invoice_roi is not None:
        # Recognize
        recognizer = InvoiceOCR(method=args.method)
        results = recognizer.recognize(invoice_roi)
        draw_texts(invoice_roi, results)
        out_path = os.path.splitext(os.path.basename(img_path))[0] + '_out.png'
        cv2.imwrite(out_path, invoice_roi)
    else:
        print('Not found any invoice')


if __name__ == '__main__':
    main()