import numpy as np
import cv2
from matplotlib import pyplot as plt
from collections import deque  
import os
import subprocess

import anvil.server
import anvil.media

with open('secrets/anvil_connection_key.txt', 'r') as file:
    connection_key = file.read()
anvil.server.connect(connection_key)

def compile_asymptote(input_file):
    if not os.path.exists(input_file):
        raise IOError("File not found: " + file)
        
    output_file = input_file[:-3] + "png"
    
    process = subprocess.Popen(["asy", "-noView", "-f", "png",
                                 "-o", output_file, input_file],
                                stderr=subprocess.PIPE)
    code = process.wait()
    if code != 0:
        raise RuntimeError(str(asy_proc.stderr.read()))

@anvil.server.callable
def image_to_asymptote(file):
    with anvil.media.TempFile(file) as filename:
        img = cv2.imread(filename, 0)
        rows = img.shape[0]
        cols = img.shape[1]
        
        # MinMax Scaler
        minPixel = np.amin(img)
        maxPixel = np.amax(img)
        img = (img - minPixel) / (maxPixel - minPixel)               
        
        # Breadth-first search implementation
        def bfs(img, visited, lines, row, col, threshold):
            queue = deque([(row,col)])
            while (queue):
                i, j = queue.popleft()
                visited[i][j] = True
                for horizontal in [-1, 0, 1]:
                    for vertical in [-1, 0, 1]:
                        if ((i + horizontal >= 0 and i + horizontal < rows) and (j + vertical >= 0 and j + vertical < cols) and img[i + horizontal][j + vertical] <= threshold and not visited[i + horizontal][j + vertical]):
                            lines.append([(j, rows - i),(j + vertical, rows - (i + horizontal))])
                            queue.append((i + horizontal, j + vertical))
                            visited[i + horizontal][j + vertical] = True 
                    
        # Populate lines of asymptote code using BFS 
        visited = np.array([[False for i in range (cols)] for j in range (rows)])
        threshold = 0.5
        lines = []
        for i in range (rows):
            for j in range (cols):
                if (not visited[i][j] and img[i][j] <= threshold):
                    bfs(img, visited, lines, i, j, threshold)
                visited[i][j] = True
        
        # Write the asymptote code to a file
        asy_code = ""
        for line in lines:
            asy_code += "draw(" + str(line[0]) + "--" + str(line[1]) + ");\n"
        with open("output.asy", "w") as output:
            output.write(asy_code)
        
        compile_asymptote("output.asy")
        asy_code_compiled = anvil.media.from_file("output.png")
            
        return asy_code, asy_code_compiled
        
anvil.server.wait_forever()