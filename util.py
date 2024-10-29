import random 

def partition_list(lst, min_size=1):
    """Randomly split the list into subsets of some minimum size."""
    
    total_length = len(lst)
    splits = []
    
    while total_length > 0:
        subset_size = random.randint(min_size, total_length)
        splits.append(lst[:subset_size])
        lst = lst[subset_size:]
        total_length -= subset_size
        
    return splits