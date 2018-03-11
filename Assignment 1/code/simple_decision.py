def predict(Xrow):
    if Xrow[1] > 37.695206:
        if Xrow[0] > -96.032692:
            y = 0
        else: 
            y = 1
    elif Xrow[0] > -113.0:
        y = 1
    else:
        y = 0
    return y