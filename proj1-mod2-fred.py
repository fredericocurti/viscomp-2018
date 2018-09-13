# Frederico Curti
import cv2 as cv
import numpy as np
import matplotlib as mpl
import time
import random
import imutils
import math

cv2 = cv
mpl.use('Qt5Agg',warn=False, force=True) # mac problems
print(cv.__version__)


def draw_square(x, y, a):
    # top-left, top-right, bottom-right, bottom-left, center
    h = int(a/2)
    return [(x-h, y-h), (x+h, y-h), (x+h, y+h), (x-h, y+h), (x,y)]


def colors(n):
  ret = []
  r = int(random.random() * 256)
  g = int(random.random() * 256)
  b = int(random.random() * 256)
  step = 256 / n
  for i in range(n):
    r += step
    g += step
    b += step
    r = int(r) % 256
    g = int(g) % 256
    b = int(b) % 256
    ret.append((r, g, b))
  return ret

# Abre uma imagem em gray scale
captura = cv2.VideoCapture(0)
# captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Parametriza a funcao do OpenCV
dt_params = dict(maxCorners=100,
                 qualityLevel=0.3,
                 minDistance=7,
                 blockSize=7)

# Parametriza o Lucas-Kanade
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Gera cores de forma aleatória
color = np.random.randint(0, 255, (100, 3))

sqr = draw_square(640, 360, 200)

## PREPARE FIRST FRAME
ret, frame = captura.read()
rows, cols = frame.shape[:2]

previous = frame
prev = cv.cvtColor(previous, cv.COLOR_BGR2GRAY)

es = 10  # edge size
edges_p = [prev[0:es, 0:es], prev[0:es, cols-es:cols],
           prev[rows-es:rows, cols-es:cols], prev[rows-es:rows, 0:es]]


# Cria uma máscara para imprimir o rastro.
mask = np.zeros_like(previous)
dx = 0
dy = 0
pdx = 0
pdy = 0

in_bounds = True

edges_avg_x = [0,0,0,0]
edges_avg_y = [0,0,0,0]

while(1):
    ret, frame = captura.read()
    actual = frame

    next = cv.cvtColor(actual, cv.COLOR_BGR2GRAY)

    sqr = draw_square(int(640 + dx), int(360 + dy), 300)


    # Farneback
    flow = cv.calcOpticalFlowFarneback(
        prev[sqr[0][1]:sqr[3][1], sqr[0][0]:sqr[1][0]], next[sqr[0][1]:sqr[3][1], sqr[0][0]:sqr[1][0]], None,  0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Não deu tempo pra rotação ;(

    # edges = [next[0:es,0:es], next[0:es,cols-es:cols], next[rows-es:rows, cols-es:cols], next[rows-es:rows,0:es]]
    # tangents = []

    # for i in range(len(edges)):
    #     ef = cv.calcOpticalFlowFarneback(
    #         edges_p[i], edges[i], None,  0.5, 3, 15, 3, 5, 1.2, 0
    #     )
    #     edges_avg_x[i] = np.average(ef[:, :, 0]) 
    #     edges_avg_y[i] = np.average(ef[:, :, 1])
    #     tangents.append(edges_avg_x[i]/edges_avg_y[i])
    
    # des = []
    # for i in range(len(edges_avg_x)):
        # print(i,math.degrees(math.atan(tangents[i])))
        # des.append((edges_avg_x[i]**2 + edges_avg_y[i]**2)**0.5)

    
    # print(math.atan2(targetY-gunY, targetX-gunX))
    # print(edges_avg_y)

    dx = np.average(flow[:,:, 0]) + pdx 
    dy = np.average(flow[:,:, 1]) + pdy
    
    img = actual

    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    dst = cv2.warpAffine(img, M, (cols, rows))

    img_rect = cv2.rectangle(dst, sqr[0], sqr[2], (0, 255, 0) if in_bounds else (0,0,255), 2)

    # Scaling
    cs = (dx**2+dy**2)**0.5
    cs = cs/100
    s = 1 + cs
    res = cv2.resize(img_rect, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    c = sqr[4]
    newx = int(c[0]*s)
    newy = int(c[1]*s)
    res = res[newy - 360 : newy + 360, newx - 640 : newx + 640]

    # usado p/ ver se esta no limite de mostar a barra preta - hardcoded
    yd = cols - c[1]
    
    # check for black bar bounds
    if (yd > 980 or yd < 860):
        in_bounds = False
    else:
        in_bounds = True
        pdx = dx
        pdy = dy
    
    try:
        cv2.imshow('image', res)
    except:
        pass

    prev = next.copy()
    edges_p = [prev[0:es, 0:es], prev[0:es, cols-es:cols],
               prev[rows-es:rows, cols-es:cols], prev[rows-es:rows, 0:es]]


    
    # Pressione ESC para sair do loop e R para resetar o tracking
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    if k == ord('r'):
        dx = 0
        dy = 0
        pdx = 0
        pdy = 0
        in_bounds = True

captura.release()
cv2.destroyAllWindows()
