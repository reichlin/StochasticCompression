



def conv(w, k, p, s):

    o = (w - k + (p+p-1)) / s + 1

    print(o)

    return int(o)


def deconv(w, k, p, s):

    o = s * (w - 1) + k - 2*p#(p+p-1)

    print(o)

    return int(o)



x = 768#168#200#512#768
print(x)

#x = x+1

print()
y = conv(x, 5, 2, 2)
y = conv(y, 5, 2, 2)
#y = conv(y, 5, 2, 2)
#y = conv(y, 3, 1, 2)
#y = conv(y, 3, 1, 2)
print("-> ", end='')
y = conv(y, 5, 2, 2)

# print()
# y_t = conv(y, 3, 1, 1)
# y_t = conv(y_t, 3, 1, 1)
# y_t = conv(y_t, 3, 1, 1)
# print()


y = deconv(y, 6, 2, 2)
#y = deconv(y, 3, 1, 2)
#y = deconv(y, 3, 1, 2)
#y = deconv(y, 5, 2, 2)
y = deconv(y, 6, 2, 2)
y = deconv(y, 6, 2, 2)
print()

#print(y-1)
#print(y)





