# Tangram Solver
This repo is the first project assignment for 2019 Autumn class: An Introduction to Artificial Intelligence, Department of Automation, Tsinghua University. A Tangram Puzzle Solver. In short, it's a project searching for a valid solution for a given Tangram puzzle using the searching algorithm we learned in the course. I've implemented three types of Tangram puzzle: classic 7 elements puzzle, interesting 13 elements puzzle and novel 7 elements puzzle with any-shape input image.  
  
<img src="https://github.com/Wuziyi616/Artificial_Intelligence_Project1/blob/master/images/UI.png" width=600 alt="UI">  
  
## Quick Start
To give you a quick start to this project, I advise you to directly run the code and play the game by yourself! I bet it will be funny. The usage is quite easy by:  
  
    $ python3 tangram_solver.py  
  
So now let me first introduce you to the UI of the game.  
  
The functions of the controls are described below.
- You can change the type of the puzzle (normal 7 elements, interesting 13 elements and novel any input) under Game Mode
- You can change the searching algorithm I implemented under Search Method (actually only DFS is available)
- If you select an index of image via from dataset, then the project will load a pre-prepared puzzle image from my own dataset
- If you want to test the project with your own image, you can specify the root of that image and use load from path to load it
- Note that you should always input the total area of the Tangram elements used in the puzzle since we need it calculate a ratio between input image and standard unit_length of Tangram elements. For example, you should always set it as 8 in a typical 7 elements puzzle
- After set the above values correctly, click Load and Solve it! The solution will be displayed on the screen and also saved to result/
- Some puzzles may be very hard to accomplish so it's possible to wait for minutes until it succeeds
  
## About the code
### Package Prerequisites
The code is tested under python3.5 and 3.6. And the UI is developed under pyqt5. There are also some other python packages needed.  
- skimage, cv2, PIL, matplotlib for image processing
- numpy for matrix calculation
  
### Algorithms
The detailed methods and algorithms I used can be found in report.pdf (written in Chinese). Here I will just give you a brief discuss.
  
#### States and Transfer Function
We should always define **STATE** and **TRANSFER FUNCTION** in a typical searching problem. In this project, the state is a **mask**, on which the areas needs to filled by elements has value 0, the areas can't be filled by elements has value 255. The transfer function is **placing one element on the mask and update the mask**.  
That is to say, when I have an input image, I initialize a mask with value 0 and 255 according to it. Then every time I place an element on a valid position **A**, I will update the mask by setting that specific area **A** with value 255. So next time when I search through the mask, I will just look at the remaining places.
  
#### Why I use only DFS?
It's been noted above that I only use DFS for searching. However, the professor delivered the class spent lots of time discussing A* algorithms and many other advanced one. So the reason I don't use those algorithms is that, from my perspective, in Tangram puzzles, every **VALID** solution is **OPTIMAL**, we only want to find one valid solution as quickly as possible, and we don't worry about sub-optimal. In that case, DFS is a simple but useful selection. Besides, it's quiet hard, if not impossible, to design an **evaluation function** for every state.
  
#### 7 Elements Puzzle
In this mode, I will first detect the corners and segments of the **mask**, and place all 7 elements in a pre-defined order. That's because we have prior knowledge that, in a 7 elements Tangram, it's impossible that both large Triangles are all surrounded by other smaller elements, at least one of it will reach the corners of the image. So we can search by the order of Large_Triangle --> Parallelogram --> Square --> Medium Triangle --> Small Triangle. This order will significantly reduce searching space since large ones are placed first.
  
#### 13 Elements Puzzle
In 13 elements puzzle, we can see that all the elements are composed of small 1x1 squares. This inspires me to segment the puzzle-image into 1x1 grid. After that, we no longer need to worry about error caused by non-integer coordinates or thing like that. Every time I place an element on the grid and update it.  
It's worthy noting that I use another pruning mechanism. I discover that all the elements have an area of 5 (except for square has 4). So I always place the square first and then, all the remaining connected area of the **mask** should have an area that is a multiple of 5! This can help me prune those states having connected area whose area is not 5 or 10 or ...
  
#### Any Input Puzzle
I've to admitted that the code didn't perform well in this puzzle. Actually in this puzzle, there may exist many valid solutions so A* algorithm could be better but I just can't find a suitable evaluation function. Maybe works on it as TODO later~  
Back to this, I copy most of the code from classic 7 elements puzzle and relax the valid-position judgment for placing elements. I make the element_out_of_mask judgment less strict but tighten the element_intersect_element judgment. Also, I don't just return the first found valid solution but the one that has the **max IoU** with the input image within the first **K** (set as 10 in the code) valid solutions.  
  
## Results
### 7 Elements Puzzle
<img src="https://github.com/Wuziyi616/Artificial_Intelligence_Project1/blob/master/images/solved_7elements.png" width=600 alt="UI">  
  
### 13 Elements Puzzle
<img src="https://github.com/Wuziyi616/Artificial_Intelligence_Project1/blob/master/images/solved_13elements.png" width=600 alt="UI">  
  
### Any Input Puzzle
<img src="https://github.com/Wuziyi616/Artificial_Intelligence_Project1/blob/master/images/solved_anyinput.png" width=600 alt="UI">  
  
## Author
Ziyi Wu  
wuzy17@mails.tsinghua.edu.cn  
