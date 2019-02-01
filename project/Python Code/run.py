from proj1_helpers import *
from implementations import *
import numpy as np

def main():
    # load data
    y, tx, ids = load_csv_data("data/train.csv", sub_sample=False)
    y_test, tx_test, ids_test = load_csv_data("data/test.csv", sub_sample=False)
    
    # Manually cluster the train and test data into four instances
    y_0, tx_0, y_1, tx_1, y_2, tx_2, y_3, tx_3, _, _, _, _ = data_segmentation(y, tx)
    y_0te, tx_0te, y_1te, tx_1te, y_2te, tx_2te, y_3te, tx_3te,\
     ind0, ind1, ind2, ind3 = data_segmentation(y_test, tx_test)
    
    """ PREPARE TRAIN DATA AND DEVELOP MODEL """
    # degree 5 polynomial expansion
    degree = 5
    tx_0 = build_poly(tx_0, degree)
    tx_1 = build_poly(tx_1, degree)
    tx_2 = build_poly(tx_2, degree)
    tx_3 = build_poly(tx_3, degree)
    
    # features to remove in each of the four instances found by backward selection
    removed_0 = [1, 50, 5, 29, 37, 80, 55]
    removed_1 = [12]
    removed_2 = [114, 95, 46, 43, 53, 74, 45, 28]
    removed_3 = [43, 128, 21, 41, 102, 54, 44, 42, 92, 20]
    
    # remove features as discovered by backward selection
    for i,c in enumerate(removed_0): tx_0 = np.delete(tx_0, c, axis=1)
    for i,c in enumerate(removed_1): tx_1 = np.delete(tx_1, c, axis=1)
    for i,c in enumerate(removed_2): tx_2 = np.delete(tx_2, c, axis=1)
    for i,c in enumerate(removed_3): tx_3 = np.delete(tx_3, c, axis=1)
    
    # Interaction terms to add in each of the four instances found by forward selection
    interaction_terms0 = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 8), (0, 9), (0, 10), (0, 14), (0, 16), (0, 20),\
                           (0, 24), (0, 25), (0, 26), (1, 6), (1, 7), (1, 10), (1, 12), (1, 16), (1, 29), (2, 9),\
                            (2, 16), (2, 19), (2, 22), (3, 5), (3, 9), (3, 14)]
    interaction_terms1 = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 6), (0, 7), (0, 8), (0, 10), (0, 12), (0, 19), (0, 22), (0, 23), (0, 24), (0, 25), (0, 27),\
                          (0, 28), (0, 29), (1, 2), (1, 3), (1, 4), (1, 8), (1, 12), (1, 14), (1, 15), (1, 18), (1, 19), (1, 21), (1, 23), (1, 24),\
                          (1, 25), (1, 26), (1, 27), (1, 29), (2, 4), (2, 7), (2, 8), (2, 10), (2, 11), (2, 18), (2, 21), (2, 25), (2, 27), (3, 8), (3, 9),\
                           (3, 12), (3, 21), (3, 22), (3, 23), (3, 25), (3, 26), (4, 6), (4, 9), (4, 10), (4, 11), (4, 12), (4, 17), (4, 22), (4, 24), (4, 25),\
                            (4, 26), (4, 27), (5, 6), (5, 8), (5, 17), (5, 24), (5, 26), (5, 29), (6, 8), (6, 21), (6, 25), (6, 28), (7, 17), (7, 19), (7, 21),\
                             (7, 22), (7, 25), (7, 28), (7, 29), (8, 12), (8, 13), (8, 14), (8, 18), (8, 19), (8, 24), (8, 29), (9, 23), (10, 18), (10, 21),\
                              (12, 23), (12, 27), (14, 15), (14, 23), (14, 26)]
    interaction_terms2 = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 8), (0, 10), (0, 14), (0, 18), (0, 29), (1, 2), (1, 3), (1, 4), (1, 7), (1, 10),\
                           (1, 11), (1, 22), (1, 26), (1, 28), (2, 3), (2, 6), (2, 7), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 17), (2, 28), (3, 4),\
                            (3, 24), (3, 28), (3, 29), (4, 5), (4, 6), (4, 7), (4, 9), (4, 11), (4, 12), (4, 15), (4, 18), (4, 21), (4, 28), (5, 16), (5, 23),\
                             (6, 8), (6, 22), (7, 10), (7, 11), (7, 29), (8, 9), (8, 14), (8, 16), (8, 19), (8, 25), (9, 17), (10, 13), (11, 15), (11, 27),\
                              (14, 18), (18, 28), (19, 23), (25, 26)]
    interaction_terms3 = [(0, 1), (0, 2), (0, 4), (0, 8), (0, 16), (0, 25), (1, 7), (1, 11), (1, 13), (1, 14), (1, 18), (2, 18),\
                           (2, 19), (2, 21), (2, 29), (3, 5), (4, 28), (6, 22), (6, 25), (10, 28), (13, 24), (14, 15), (20, 26), (20, 28)]    
    
    # add 2nd order interaction terms
    for i, e in enumerate(interaction_terms0): 
        tx_0 = np.c_[tx_0, tx_0[:,e[0]] * tx_0[:,e[1]]]
    for i, e in enumerate(interaction_terms1): 
        tx_1 = np.c_[tx_1, tx_1[:,e[0]] * tx_1[:,e[1]]]
    for i, e in enumerate(interaction_terms2): 
        tx_2 = np.c_[tx_2, tx_2[:,e[0]] * tx_2[:,e[1]]]
    for i, e in enumerate(interaction_terms3): 
        tx_3 = np.c_[tx_3, tx_3[:,e[0]] * tx_3[:,e[1]]]
    
    # 3rd order Interaction terms to add in each of the four instances found by forward selection
    third_interaction_terms0 = [(0, 2, 15), (2, 6, 23), (3, 4, 10), (3, 4, 21), (3, 9, 25), (3, 11, 24), (3, 17, 19), (3, 21, 28), (3, 22, 24),\
                                  (3, 23, 24), (3, 25, 28), (3, 27, 28), (4, 5, 6), (4, 5, 8), (4, 5, 14), (4, 5, 18), (4, 6, 12), (4, 6, 16),\
                                   (4, 6, 18), (4, 6, 23), (4, 9, 22), (4, 9, 26), (4, 11, 23), (4, 16, 24), (5, 14, 19), (5, 14, 23), (5, 16, 23),\
                                    (5, 18, 26), (5, 19, 26), (5, 20, 25), (5, 22, 26), (6, 9, 13), (6, 11, 16), (6, 15, 25), (6, 15, 28), (6, 17, 29),\
                                     (6, 18, 27), (7, 9, 28), (7, 9, 29), (7, 10, 17), (7, 10, 22), (7, 11, 14), (7, 11, 17), (7, 14, 17), (11, 17, 21),\
                                      (11, 19, 23), (12, 26, 29), (13, 18, 19), (13, 18, 21), (17, 19, 21)]
    third_interaction_terms1 = [(0, 1, 4), (0, 1, 10), (0, 1, 22), (0, 1, 23), (0, 1, 24), (0, 1, 26), (0, 1, 28), (0, 2, 3), (0, 2, 4), (0, 2, 26),\
                                  (0, 2, 28), (0, 3, 4), (0, 3, 16), (0, 3, 25), (0, 4, 8), (0, 4, 9), (0, 4, 11), (0, 4, 25), (0, 4, 26), (0, 4, 28),\
                                   (0, 5, 6), (0, 5, 9), (0, 5, 21), (0, 5, 25), (0, 5, 28), (0, 6, 18), (0, 6, 24), (0, 6, 26), (0, 8, 12), (0, 8, 28),\
                                    (0, 11, 26), (0, 12, 21), (0, 12, 29), (0, 13, 14), (0, 13, 25), (0, 14, 26), (0, 14, 28), (0, 15, 16), (0, 15, 21),\
                                     (0, 16, 26), (1, 2, 5), (1, 12, 15), (1, 15, 25), (3, 13, 25), (4, 13, 21), (9, 18, 26), (10, 13, 26), (11, 15, 26),\
                                      (11, 18, 22), (14, 18, 27), (15, 19, 23), (17, 22, 25), (21, 22, 23)]
    third_interaction_terms2 = [(0, 1, 7),(0, 1, 8),(0, 1, 9),(0, 1, 13),(0, 1, 14),(0, 1, 15),(0, 1, 16),(0, 1, 18),(0, 1, 19),(0, 1, 21),(0, 1, 28),\
                                 (0, 1, 29),(0, 2, 7),(0, 2, 8),(0, 2, 10),(0, 2, 14),(0, 2, 16),(0, 2, 18),(0, 2, 20),(0, 2, 25),(0, 2, 28),(0, 3, 4),\
                                  (0, 3, 7),(0, 3, 8),(0, 3, 9),(0, 3, 10),(0, 3, 12),(0, 3, 21),(0, 3, 23),(0, 4, 28),(0, 5, 17),(0, 5, 26),(0, 6, 11),\
                                  (0, 6, 27),(0, 8, 13),(0, 8, 14),(0, 9, 18),(0, 10, 17),(0, 12, 24),(0, 13, 16),(0, 13, 27),(0, 16, 17),(1, 2, 18),\
                                  (1, 17, 27),(2, 20, 24),(3, 5, 14),(3, 7, 17),(4, 10, 26),(5, 17, 18),(5, 17, 26),(7, 9, 12),(7, 9, 17),(7, 9, 29),\
                                  (7, 10, 13),(7, 10, 28),(7, 11, 23),(7, 11, 26),(7, 11, 28),(7, 11, 29),(7, 12, 16),(7, 13, 25),(7, 14, 21),(7, 15, 18),\
                                  (7, 15, 27),(7, 16, 17),(7, 16, 25),(7, 18, 23),(7, 20, 28),(7, 20, 29),(7, 21, 25),(7, 24, 29),(7, 25, 27),(8, 10, 19),\
                                  (8, 14, 17),(8, 18, 25),(9, 10, 20),(9, 15, 25),(9, 18, 25),(10, 14, 29),(10, 20, 24),(10, 22, 24),(11, 14, 15),(11, 15, 17),\
                                  (11, 15, 19),(11, 16, 24),(11, 17, 24),(11, 17, 25),(11, 18, 19),(11, 23, 28),(12, 14, 15),(12, 16, 26),(14, 18, 21),\
                                  (16, 23, 29),(17, 18, 29)]
    third_interaction_terms3 = [(0, 1, 4), (0, 1, 7), (0, 1, 9), (0, 1, 10), (0, 1, 11), (0, 1, 12), (0, 1, 20), (0, 1, 24), (0, 1, 27), (0, 2, 3),\
                                  (0, 2, 7), (0, 2, 10), (0, 2, 11), (0, 2, 17), (0, 2, 24), (0, 2, 29), (0, 3, 5), (0, 3, 7), (0, 3, 9), (0, 3, 15),\
                                   (0, 3, 23), (0, 3, 24), (0, 4, 5), (0, 4, 12), (0, 4, 13), (0, 4, 14), (0, 4, 15), (0, 4, 20), (0, 4, 21), (0, 4, 22),\
                                    (0, 5, 13), (0, 6, 16), (0, 9, 16), (0, 9, 19), (0, 9, 20), (0, 14, 25), (1, 4, 22), (1, 5, 8), (1, 6, 22), (1, 8, 14),\
                                     (1, 9, 18), (1, 10, 21), (1, 17, 24), (2, 4, 10), (2, 4, 21), (2, 20, 21), (4, 10, 12), (4, 10, 13), (4, 10, 15),\
                                      (4, 10, 17), (4, 11, 15), (4, 16, 17), (4, 18, 26), (4, 25, 26), (4, 25, 29), (5, 19, 21), (7, 22, 28), (8, 9, 23),\
                                       (8, 9, 27), (10, 11, 24), (12, 22, 28), (13, 16, 20), (14, 19, 25), (16, 23, 28)]
    
    # add 3rd order interaction terms
    for i, e in enumerate(third_interaction_terms0): 
        tx_0 = np.c_[tx_0, tx_0[:,e[0]] * tx_0[:,e[1]] * tx_0[:,e[2]]]
    for i, e in enumerate(third_interaction_terms1): 
        tx_1 = np.c_[tx_1, tx_1[:,e[0]] * tx_1[:,e[1]] * tx_1[:,e[2]]]
    for i, e in enumerate(third_interaction_terms2): 
        tx_2 = np.c_[tx_2, tx_2[:,e[0]] * tx_2[:,e[1]] * tx_2[:,e[2]]]
    for i, e in enumerate(third_interaction_terms3): 
        tx_3 = np.c_[tx_3, tx_3[:,e[0]] * tx_3[:,e[1]] * tx_3[:,e[2]]]
        
    # develop model
    w0, _ = least_squares(y_0, tx_0)
    w1, _ = least_squares(y_1, tx_1)
    w2, _ = least_squares(y_2, tx_2)
    w3, _ = least_squares(y_3, tx_3)
    
    
    """ PREPARE TEST DATA AND MAKE PREDICTIONS """
    # degree 5 polynomial expansion
    tx_0te = build_poly(tx_0te, degree)
    tx_1te = build_poly(tx_1te, degree)
    tx_2te = build_poly(tx_2te, degree)
    tx_3te = build_poly(tx_3te, degree)
    
    # remove features as discovered by backward selection
    for i,c in enumerate(removed_0): tx_0te = np.delete(tx_0te, c, axis=1)
    for i,c in enumerate(removed_1): tx_1te = np.delete(tx_1te, c, axis=1)
    for i,c in enumerate(removed_2): tx_2te = np.delete(tx_2te, c, axis=1)
    for i,c in enumerate(removed_3): tx_3te = np.delete(tx_3te, c, axis=1)
    
    # add interaction terms
    for i, e in enumerate(interaction_terms0): 
        tx_0te = np.c_[tx_0te, tx_0te[:,e[0]] * tx_0te[:,e[1]]]
    for i, e in enumerate(interaction_terms1): 
        tx_1te = np.c_[tx_1te, tx_1te[:,e[0]] * tx_1te[:,e[1]]]
    for i, e in enumerate(interaction_terms2): 
        tx_2te = np.c_[tx_2te, tx_2te[:,e[0]] * tx_2te[:,e[1]]]
    for i, e in enumerate(interaction_terms3): 
        tx_3te = np.c_[tx_3te, tx_3te[:,e[0]] * tx_3te[:,e[1]]]
    
    # add 3rd order interaction terms
    for i, e in enumerate(third_interaction_terms0): 
        tx_0te = np.c_[tx_0te, tx_0te[:,e[0]] * tx_0te[:,e[1]] * tx_0te[:,e[2]]]
    for i, e in enumerate(third_interaction_terms1): 
        tx_1te = np.c_[tx_1te, tx_1te[:,e[0]] * tx_1te[:,e[1]] * tx_1te[:,e[2]]]
    for i, e in enumerate(third_interaction_terms2): 
        tx_2te = np.c_[tx_2te, tx_2te[:,e[0]] * tx_2te[:,e[1]] * tx_2te[:,e[2]]]
    for i, e in enumerate(third_interaction_terms3): 
        tx_3te = np.c_[tx_3te, tx_3te[:,e[0]] * tx_3te[:,e[1]] * tx_3te[:,e[2]]]
    
    # make predictions!
    y_pred0 = predict_labels(w0, tx_0te)
    y_pred1 = predict_labels(w1, tx_1te)
    y_pred2 = predict_labels(w2, tx_2te)
    y_pred3 = predict_labels(w3, tx_3te)
    
    # consolidate predictions
    y_pred = np.empty(len(y_test))
    y_pred[ind0] = y_pred0
    y_pred[ind1] = y_pred1
    y_pred[ind2] = y_pred2
    y_pred[ind3] = y_pred3
    
    # create csv submission file
    create_csv_submission(ids_test, y_pred, "submission_optimal.csv")


if __name__ == "__main__": main()