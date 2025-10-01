# Profiles
## Description
Un Profile de couleur est la moyenne des canaux LAB et leurs deviations calculee a partir des flux de chaque annee de SPOT de 2013 vers 2024 et un autre calculee apartire de la moyenne de tous profiles millesimaux 

## Utilisation 
``` python
# definir la fonction de color_transfer
# la source est une image RGB ou BGR a specifier
# LAB_means et LAB_stds : ce sont les parametres du profile choisi

def color_transfer(source, LAB_means, LAB_stds, BGR = False ):
    """
    Transfers color distribution from the target to the source.
    """
    # Convert images to LAB color space
    src_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    if BGR :
        src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_src, a_src, b_src = cv2.split(src_lab)
    
    
    # Mean and std for each channel of source
    src_mean, a_src_mean, b_src_mean = src_lab.mean(axis=(0,1))
    src_std, a_src_std, b_src_std = src_lab.std(axis=(0,1))
    
    # Mean and std for each channel of target
    tar_mean, a_tar_mean, b_tar_mean = LAB_means
    tar_std, a_tar_std, b_tar_std = LAB_stds

    # Perform color transfer
    l_new = (l_src - src_mean) * (tar_std / src_std) + tar_mean
    a_new = (a_src - a_src_mean) * (a_tar_std / a_src_std) + a_tar_mean
    b_new = (b_src - b_src_mean) * (b_tar_std / b_src_std) + b_tar_mean
    
    # Clip to valid range
    l_new = np.clip(l_new, 0, 255)
    a_new = np.clip(a_new, 0, 255)
    b_new = np.clip(b_new, 0, 255)

    # Combine back into LAB
    transfer_lab = cv2.merge([l_new, a_new, b_new]).astype(np.uint8)

    # Convert back to RGB
    transfer = cv2.cvtColor(transfer_lab, cv2.COLOR_LAB2RGB)
    if BGR:
        transfer = cv2.cvtColor(transfer_lab, cv2.COLOR_LAB2BGR)

    return transfer

# exemple d'utilisation
year = 2015 # [2013 .. 2024]
profile = np.load(f'target_profile_{year}.npz')
lab_means, lab_stds = profile['lab_means'], profile['lab_stds']

img_fp = "path/to/your/image"
image = cv2.imread(img_fp)# returns a BGR image

res = color_transfer(img, lab_means, lab_stds, BGR=True)
```