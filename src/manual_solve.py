#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

'''
Student name - Andrew Neary
ID number - 19240310
Github repository URL - https://github.com/aneary12/ARC

For all of my solve_* functions, I only used the numpy library and its 
functions. numpy.shape() is used in all functions, and numpy.zeros() is used to 
initialise empty arrays that are then usually populated by iterating through 
for loops. Indexing and slicing is used to manipulate arrays. The outputs of 
all the solve_* functions are cast to integers using the inbuilt python 
.astype() so that the outputs match the test outputs.

The first (137eaa0f) and second (41e4d17e) tasks are somewhat 
similar in that they both involve locating regions of interest within the input 
grid. As the first task is only locating a single cell, numpy.where() is 
sufficient. However, as the region of interest is more complex in the second 
task, a function is written to search through the input. This function operates 
somewhat similarly to the convolution operation used in CNNs, by sliding a 
kernel across all possible positions in the input. However, rather than perform 
a calculation, the function checks if the sub-region of the input matches the 
kernel using the numpy.all() function.

The third task was chosen as it is quite different to the first two tasks. 
Whereas the first two tasks involved searching for a region of interest in the 
input, the third task simply required a manipulation of the entirety of the 
input. The solution for the third task required a re-framing of the problem 
from what is intuitive to the human brain (colours moving inwards one step), 
to one that is easily handled computationally (considering the areas occupied 
by each colour).
'''

def solve_137eaa0f(x):
    
    '''
    For this task, the input is an 11x11 grid (default colour is black). 
    On the grid are a number of grey squares, each surrounded by a number of 
    coloured squares.
    The output is a 3x3 grid, with a grey square as the centre, and the 
    surrounding squares are coloured so that they match the pattern of coloured 
    squares around the grey squares in the input (any squares not assigned a 
    colour remain black).
    
    This function solves this task first by finding the location of the grey 
    squares in the input.
    The 3x3 grid of squares surrounding each grey square is then found.
    These 3x3 grid of squares are stored in a 3-dimensional numpy array (of 
    shape (3,3,n)).
    Because no colours ever overlap, the colour for the output is calculated by 
    taking the maximum value in the 3rd dimension of the array.
    
    All training and test examples are solved correctly.
    '''
    
    # Make sure the input is a numpy array
    inp = np.array(x)
    
    # Find the locations of grey squares in the input
    grey_squares = np.where(inp==5)
    
    # Find how many grey squares there are
    num_greys = np.shape(grey_squares)[1]
    
    # Create a 3d numpy array
    threeD_output = np.zeros((3,3,num_greys))
    
    # Iterate through the grey squares to construct a 3x3 grid with grey in the centre
    for i in range(num_greys):
    
        # Find location of grey square
        loc_x, loc_y = grey_squares[0][i], grey_squares[1][i]

        # Construct a 3x3 matrix around the grey square
        threeXthree = inp[(loc_x-1):(loc_x+2), (loc_y-1):(loc_y+2)]
    
        # Save threeXthree as layer in threeD_output
        threeD_output[:,:,i] = threeXthree
    
    # Find the final output by taking the max of each square surrounding the grey centre
    ans = np.max(threeD_output, axis=2)
    
    # Return the answer
    return ans.astype(int)

        
def solve_41e4d17e(x):
    
    '''
    For this task, the input is a 15x15 grid (default colour is light blue).
    On this grid, there are one or more dark blue 5x5 squares, where the
    central 3x3 square remains light blue (i.e. only the perimeter of the 
    square is coloured in).
    The output is created by drawing a pink cross (i.e. a straight line 
    horizontally and vertically) that passes through the centre of the 5x5 dark 
    blue square, but does not change the colour of any dark blue squares.
    
    This solution first finds the colours of the background and the squares, 
    and then constructs a 5x5 array so that it matches the 5x5 squares in the 
    input.
    Then, a function is written that acts as a sliding window over the input.
    This is to say that the sample square from above is placed over the upper 
    left most 5x5 grid in the input. 
    If  there is a match (i.e. if this 5x5 square of the input is a dark blue 
    square), this is recorded as 1, if not this is 0.
    The 5x5 sample square is then shifted to the right by one square, and so on
    across the whole top of the input.
    Once the sample square has travelled across to the upper right most corner 
    of the input, it is shifted down by one square and back to the left most
    side.
    This process is repeated across the whole input.
    The output of the sliding window function is then used to extrapolate the
    locations of the squares.
    A pink cross is draw passing through the centre of each square, and the 
    colours of the dark blue squares is re-instated afterwards.
    
    All training and test examples are solved correctly.
    '''
    
    # Make sure the input is a numpy array
    inp = np.array(x)
    
    # Find the colours in the input
    colours, hits = np.unique(inp, return_counts=True)
    square_col = colours[np.argmin(hits)]
    background_col = colours[np.argmax(hits)]

    # Create a test square
    test_square = np.ones((5,5)) * background_col
    test_square[:,0] = square_col
    test_square[:,-1] = square_col
    test_square[0,:] = square_col
    test_square[-1,:] = square_col

    # Write a function that creates a sliding window
    def sliding_window(input_array, kernel):
        # Define some important variables (same nomenclature as convolution in CNNs)
        n_in = np.shape(input_array)[0] # assume the grid is square
        k = np.shape(kernel)[0] # assume the screen is square as well
        p = 0 # no padding
        s = 1 # step size of 1
        n_out = ((n_in + 2*p - k) / s) + 1
        # Initialise variables for iterating through the input array
        output = np.zeros((int(n_out),int(n_out)))
        # Loop through the rows
        j = 0
        while j < n_out:
            # Loop through the columns
            i = 0
            while i < n_out:            
                # Check if local array is a match for the kernel
                local_array = input_array[i:(i+k),j:(j+k)]
                if np.all(local_array==kernel):
                    output[i,j] = 1
                # Move one step to the right
                i += 1
            # Move one step down
            j += 1
        # Return the output
        return output
    
    # Find the locations of the squares in the input
    square_bool = sliding_window(inp, test_square)

    # Find the co-ordinates of the top left corner of each square
    corner_locs = np.array(np.where(square_bool==1))

    # As the squares are 5x5, find the co-ordinates of the centre by adding 2 to each dimension
    centre_locs = corner_locs + 2

    # Define the number of squares in the input
    num_squares = np.shape(centre_locs)[1]

    # Draw crosses for all squares in the input
    ans = inp
    cross_col = 6 # define the colour of the cross
    for i in range(num_squares):
    
        # Get x and y co-ordinates for the square
        loc_x, loc_y = centre_locs[0][i], centre_locs[1][i]
    
        # Change colour of row
        ans[loc_x,:] = cross_col
    
        # Change colour of column
        ans[:,loc_y] = cross_col
    
        # Re-set the borders of the square
        ans[loc_x+2,loc_y] = square_col
        ans[loc_x-2,loc_y] = square_col
        ans[loc_x,loc_y+2] = square_col
        ans[loc_x,loc_y-2] = square_col
    
    # Return the answer
    return ans.astype(int)


def solve_bda2d7a6(x):
    
    '''
    For this task, the input is either a 6x6 or an 8x8 grid, where the centre 4 
    squares are one colour, the surrounding squares another colour, the squares 
    surrounding them another, and so on (like concentric squares).
    The output is constructed by shifting the second-innermost colour in to the
    middle, the third-innermost (or outside colour in the case of a 6x6 grid) 
    colour to the second-innermost, and the original centre colour to the 
    third-innermost position.
    Where the input is an 8x8 grid, the outermost colour is set to be the same 
    as the centre colour.
    
    (An alternative formulation of the problem is used for this solution.
    Instead of shifting the colours in one, we can think of the colour that 
    occupies the largest area being being mapped to the concentric square that 
    occupies the second-largest area, the colour that occupies the 
    second-largest area being mapped to the concentric square that occupies the 
    smallest area, and the colour that occupies the smallest area being mapped 
    to the concentric square that occupies the largest area.)
    
    This solution fist identifies all colours in the input and calculates the 
    area occupied by each.
    The colours are then sorted in the order of largest area to smallest.
    The colours in the input are then mapped to the correct colours for the 
    output as described above.
    
    All training and test examples are solved correctly.
    '''
    
    # Specify the input and output
    inp = np.array(x)
    
    # Find the colours in the input
    colours = np.unique(inp)
    
    # Define the number of colours in the input
    num_colours = len(colours)
    
    # Find the area that each colour takes up
    areas = np.zeros((num_colours,))
    for i in range(num_colours):
        areas[i] = np.sum(inp==colours[i])
        
    # Sort the colours from largest area to smallest
    sorted_colours = np.flip(colours[np.argsort(areas)])
    
    # Map original colours to new colours
    new_order = np.hstack((sorted_colours[-1], sorted_colours[:-1]))
    
    # Replace each colour with the appropriate new colour
    ans = np.zeros(np.shape(inp))
    for i in range(num_colours):
        ans[inp==sorted_colours[i]] = new_order[i]
    
    # Return the answer
    return ans.astype(int)


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    

def read_ARC_JSON(filepath):
    
    """
    Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output.
    """
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    
    """
    Given a task ID, call the given solve() function on every
    example in the task data.
    """
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))

if __name__ == "__main__": main()