import numpy as np

def calculate_DOS(A_d, B_d, most_likely_model_d):
    """
    A_d boolean array for label A
    B_d boolean array for label B
    most_likely_model_d integer array of class labels

    A_d, B_d and most_likely_model_d must be the same length

    Calculate the degree of separation (DOS) for labeled frames between classes:
        Let A be the set of frames with label 1
        Let B be the set of frames with label 2

        Let C_i be the set of frames which have class i as the most likely class

        C_i ∩ A : set of frames with label A in class i
        C_i ∩ B : set of frames with label B in class i

        OA_i = |C_i ∩ A| : number of frames with label A in class i

        DOS = 1 - (sum_i OA_i * OB_i) / ||OA_i|| ||OB_i||
    """
    C = np.max(most_likely_model_d)+1

    # number of frames with label a in class c
    oa_c = np.bincount(most_likely_model_d[A_d], minlength=C)
    ob_c = np.bincount(most_likely_model_d[B_d], minlength=C)

    DOS = 1 - np.sum(oa_c * ob_c) / (np.sum(oa_c**2) * np.sum(ob_c)**2)**0.5
    return DOS

