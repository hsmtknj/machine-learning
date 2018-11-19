"""
unsupervised learning algorithm

implement k-means method
"""
import numpy as np


def thresh_kmeans(img_input, itr=5, mu1_init=50, mu2_init=150):
    """
    binarize input image using k-means algorithm

        :param img_input: ndarray uint8, 1ch image
        :param itr: int, iterator of k-means algorithm
        :param mu1_init: int, initial centroid of cluster1
        :param mu2_init: int, initial centroid of cluster2
        :return img_bin: ndarray uint8, binarized image
    """

    # histogram
    hist, bins = np.histogram(img_input.ravel(), 256, [0, 256])

    # -- k-means algoritym -- #
    # initialize mean (centroid)
    mu1_current = mu1_init  # initial centroid of cluster1
    mu2_current = mu2_init  # initial centroid of cluster2

    # iterative calculation
    itr_cnt = 0  # iterate conter
    while(itr_cnt < itr):
        # initialize
        mu1_updated = 0
        mu2_updated = 0
        sum1 = 0
        sum2 = 0
        cnt1 = 0
        cnt2 = 0
        for i in range(len(hist)):
            # in the case of belonging to cluster1
            if (np.abs(i - mu1_current) < np.abs(i - mu2_current)):
                sum1 += hist[i]*i
                cnt1 += hist[i]
            # in the case of belonging to cluster2
            else:
                sum2 += hist[i]*i
                cnt2 += hist[i]
        # update each centroid
        mu1_updated = sum1 / cnt1
        mu2_updated = sum2 / cnt2
        mu1_current = mu1_updated
        mu2_current = mu2_updated
        itr_cnt += 1

    # calculate binary threshold
    thresh = (mu1_current + mu2_current) / 2
    # convert gray image to binary image
    # Note: THRESH_OTSU is also good
    max_pixel = 255
    ret, img_bin = cv2.threshold(img_input,
                                 thresh,
                                 max_pixel,
                                 cv2.THRESH_BINARY)

    return img_bin