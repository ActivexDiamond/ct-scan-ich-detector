#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 22:44:57 2022
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""
############################## Dependencies ##############################
import numpy

############################## Features - STFA ##############################
def compute_sfta(image, threshold_counts):
    """
    This function was originally written by Ebenlson.
    Big thanks to him for providing it on github.
    Original repo: https://github.com/ebenolson/tessa
    """
    
    ## Image Shape Guard    
    if len(numpy.shape(image)) == 3:
        image = numpy.mean(image, 2)
    elif len(numpy.shape(image)) != 2:
        raise "Invalid image dimension"

    ##Helpers
    def haus_dim(image):
        max_dim = numpy.max(numpy.shape(image))
        c_l_dim = numpy.ceil(numpy.log2(max_dim))
        new_dim = int(2 ** c_l_dim)
        row_pad = new_dim - numpy.shape(image)[0]
        col_pad = new_dim - numpy.shape(image)[1]
        pad = ((0, row_pad), (0, col_pad))
        image = numpy.pad(image, pad, "constant")
        
        box_counts = numpy.zeros(int(c_l_dim) + 1)
        resolutions = numpy.zeros(int(c_l_dim) + 1)
        
        image_size = numpy.shape(image)[0]
        box_size = 1
        i = 0
    
        while box_size <= image_size:
            box_count = (image > 0).sum()
            box_counts[i] = box_count
            resolutions[i] = 1.0 / box_size
            
            i += 1
            box_size *= 2
            
            image = image[::2, ::2]+image[1::2, ::2]+image[1::2, 1::2]+image[::2, 1::2]
        d = numpy.polyfit(numpy.log(resolutions), numpy.log(box_counts), 1)
        return d[0]
    #End haus_dim
    
    def find_borders(image):
        pad = [[1, 1], [1, 1]]
        i = numpy.pad(image, pad, "constant", constant_values=1).astype("uint8")
        
        i2 = i[2:, 1:-1]+i[0:-2, 1:-1]+i[1:-1:, 2:]+i[1:-1:, 0:-2] + \
            i[2:, 2:]+i[2:, 0:-2]+i[0:-2, 2:]+i[0:-2, 0:-2]
        return image * (i2 < 8)
    #End find_borders
                
    def otsu(counts):
        p = counts * 1.0 / numpy.sum(counts)
        omega = numpy.cumsum(p)
        mu = numpy.cumsum(p * range(1, len(p) + 1))
        mu_t = mu[-1]
        
        numpy.seterr(divide='ignore', invalid='ignore')
        sigma_b_squared = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
        numpy.seterr(divide='warn', invalid='warn')
        
        max_val = numpy.max(numpy.nan_to_num(sigma_b_squared))
        
        if not numpy.isnan(sigma_b_squared).all():
            return numpy.mean((sigma_b_squared == max_val).nonzero()) + 1
        return 0
    #End otsu
    
    def otsu_rec(image, total):
        if image == []:
            t = []
            return [v[0] for v in t]

        image = image.astype(numpy.uint8).flatten()
        
        NUM_BINS = 256
        counts = numpy.histogram(image, range(NUM_BINS))[0]
        
        t = numpy.zeros((total, 1))
        def otsu_rec_helper(lower_bin, upper_bin, lower_t, upper_t):
            if upper_t < lower_t or upper_bin < lower_bin:
                return
            start = int(numpy.ceil(lower_bin)) - 1
            end = int(numpy.ceil(upper_bin))
            level = otsu(counts[start:end]) + lower_bin
            
            insert_pos = int(numpy.ceil((lower_t + upper_t) / 2.0))
            t[insert_pos - 1] = level / NUM_BINS
            otsu_rec_helper(lower_bin, level, lower_t, insert_pos - 1)
            otsu_rec_helper(level + 1, upper_bin, insert_pos + 1, upper_t)
        #End otsu_rec_helper
        otsu_rec_helper(1, NUM_BINS, 1, total)
        return [v[0] for v in t]
    #End otsu_rec

    ##Main Body
    image = image.astype(numpy.uint8)
    t = otsu_rec(image, threshold_counts)
    d_size = len(t) * 6
    d = numpy.zeros(d_size)
    pos = 0
    for i in range(len(t)):
        threshold = t[i]
        ib = image > (threshold *  255)
        ib = find_borders(ib)
    
        vals = image[ib.nonzero()].astype(numpy.double)
        d[pos] = haus_dim(ib)
        pos += 1
        
        d[pos] = numpy.mean(vals)
        pos += 1
        
        d[pos] = len(vals)
        pos += 1
            
    t = t+[1.0, ]
    for i in range(len(t) - 1):
        lower_threshold = t[i]
        upper_threshold = t[i + 1]
        ib = (image > (lower_threshold * 255)) * (image < (upper_threshold * 255))
        ib = find_borders(ib)
        
        vals = image[ib.nonzero()].astype(numpy.double)
        d[pos] = haus_dim(ib)
        pos += 1
        
        d[pos]= numpy.mean(ib)
        pos += 1
        
        d[pos] = len(vals)
        pos += 1
    
    return d
