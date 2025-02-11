import numpy as np
import chess

table_dict = {
    'a' : 0,
    'b' : 1,
    'c' : 2,
    'd' : 3,
    'e' : 4,
    'f' : 5,
    'g' : 6,
    'h' : 7
}

num2files = {
    0 : 'a' ,
    1 : 'b' ,
    2 : 'c' ,
    3 : 'd' ,
    4 : 'e' ,
    5 : 'f' ,
    6 : 'g' ,
    7 : 'h'
}

def encode_move(move):
    f1 = int(table_dict[move[0]])
    f2 = int(table_dict[move[2]])

    r1 = int(move[1]) - 1
    r2 = int(move[3]) - 1

    diff_f = abs(f2-f1)
    diff_r = abs(r2-r1)

    # knight moves:
    if (diff_f == 1 and diff_r > 1) or (diff_r == 1 and diff_f > 1):
        # knight
        if diff_r == 2:
            if r2 > r1:
                if f1 > f2:
                    return 0
                else:
                    return 1
            else:
                if f1 > f2:
                    return 2
                else:
                    return 3
        else:
            if r2 > r1:
                if f1 > f2:
                    return 4
                else:
                    return 5
            else:
                if f1 > f2:
                    return 6
                else:
                    return 7
    else:

        # queen moves:
        if diff_f == 0:
            if r2 > r1:
                return diff_r + 7 # 7 as those are the first 8 planes for knight moves
                # These are the planes for moving upwards..
            else:
                # These are the planes for moving downwards..
                return diff_r + 14
        elif diff_r == 0:
            if f2 > f1:
                return diff_f + 21
                # These are the planes for moving to the right..
            else:
                # These are the planes for moving to the left..
                return diff_f + 28
        else:
            if f2 > f1 and r2 > r1:
                return diff_r + 35
            elif f2 > f1 and r2 < r1:
                return diff_r + 42
            elif f2 < f1 and r2 < r1:
                return diff_r + 49
            else: # f2 < f1 and r2 > r1:
                return diff_r + 56

# The input is the position in 12 planes and the output will be the 64 planes encoded output...
def encode_output(move,y,prob):
    
    moves_plane = encode_move(move)

    y[moves_plane,[int(move[1])-1],table_dict[move[0]]] += prob

    return y


def create_position_planes2(position):
    board = chess.Board(position)

    # Check if we need to turn the board
    turned = False
    if not board.turn:
        turned = True
        board = board.mirror()  # Use the chess library's built-in method to flip the board for black

    # Create planes for 12 types of pieces (6 for each side)
    planes = np.zeros((13, 8, 8), dtype=np.int8)

    attacked_map = attacked_squares_map(board)
    planes[0] = attacked_map
    # Mapping chess pieces to plane indices
    piece_map = {
        'K': 1, 'Q': 2, 'R': 3, 'B': 4, 'N': 5, 'P': 6,
        'k': 7, 'q': 8, 'r': 9, 'b': 10, 'n': 11, 'p': 12
    }

    # Efficient iteration over the board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)  # Get row and column from square index
            '''if turned:
                row = 7 - row  # Flip the row if board is turned for black
                col = 7 - col  # Flip the column as well'''
            
            plane_idx = piece_map[piece.symbol()]
            planes[plane_idx, row, col] = 1

    return planes, turned

def create_position_planes(position):
    board = chess.Board(position)

    # Create planes for 12 types of pieces (6 for each side)
    planes = np.zeros((14, 8, 8), dtype=np.int8)

    from_pieces = np.zeros((8,8))
    
    attacked_map = attacked_squares_map(board,board.turn)
    planes[12] = attacked_map
    # Mapping chess pieces to plane indices
    piece_map = {
        'K': 0, 'Q': 1, 'R': 2, 'B': 3, 'N': 4, 'P': 5,
        'k': 6, 'q': 7, 'r': 8, 'b': 9, 'n': 10, 'p': 11
    }

    # Efficient iteration over the board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)  # Get row and column from square index

            plane_idx = piece_map[piece.symbol()]
            planes[plane_idx, row, col] = 1
    
    if board.turn:
        from_pieces = np.sum(planes[0:6], axis=0)
    else:
        from_pieces = np.sum(planes[6:12], axis=0)
    
    planes[13] = from_pieces


    return planes

turn_dict = {
    1:8,
    2:7,
    3:6,
    4:5,
    5:4,
    6:3,
    7:2,
    8:1

}

def turn(move):
    new_best_move = []
    new_best_move.append(num2files[abs(table_dict[move[0]])])
    new_best_move.append(abs(int(move[1])-8)) # Taking one more to shift it to [0...7]
    new_best_move.append(num2files[abs(table_dict[move[2]])])
    new_best_move.append(abs(int(move[3])-8)) # Taking one more to shift it to [0...7]
    return ''.join(map(str, new_best_move))

# Change it so there is a lsit for 4 best moves and and np.array for cp s than change it to power transofrm

# k: control the decay rate
def best_move_distribution(position,side,max_moves = 4, max_cp_diff = 40, beta = 0.04, depth=18): 

    is_mate = 'mate' in position['evals'][0]['pvs'][0]

    best_move = position['evals'][0]['pvs'][0]['line'][:4]
    move_dist = []
    move_dist.append(best_move)
    
    # If its a mate, than we greedily mate, we dont look for other possible good moves.
    if is_mate:
        return move_dist, np.asarray([1.0])
    
    best_cp = int(position['evals'][0]['pvs'][0]['cp'])
    cp2probs = [] # This array is for storing best moves cp's than transforming them into probs...
    
    if best_cp == 0:
        cp2probs.append(best_cp+1)
    else:
        cp2probs.append(best_cp)
    
    for eval in position['evals']:
        
        if eval['depth'] >= depth:
            for line in eval['pvs']:

                # If there is a mate, after the move, that is an only move scenerio, so we disregard the rest...
                if 'mate' in line:
                    break

                curr_cp = int(line['cp'])
                if np.abs(curr_cp - best_cp) <= max_cp_diff:
                    if not line['line'][:4] in move_dist: 
                        move_dist.append(line['line'][:4])
                        if curr_cp == 0:
                            cp2probs.append(curr_cp+1) # we add 1 to all, so when the position is equal we can still transform it
                        else:
                            cp2probs.append(curr_cp)
            
                if len(move_dist) == max_moves:
                    break

        if len(move_dist) == 4:
                break
    
    cp2probs = np.asarray(cp2probs)
    
    # Softmax with temperature beta:
    min_cp_loss = np.min(cp2probs)
    normalized_losses = np.array(cp2probs) - min_cp_loss
    
    # if side is true its white turn so we need to find the highest cp else its black and we are searching for the lowest:
    if side:
        exp_values = np.exp(beta * normalized_losses)
    else:
        exp_values = np.exp(-beta * normalized_losses)
    
    # Normalize to get probabilities
    probabilities = exp_values / np.sum(exp_values)
    
    return move_dist, probabilities
    
# This funciton creates an attacked_by binary map
def attacked_squares_map(board, color=chess.WHITE):
    # Initialize an 8x8 grid of zeros
    attacked_map = np.zeros((8, 8), dtype=int)
    
    # Loop over all squares on the board (0 to 63)
    for square in chess.SQUARES:
        # Check if the square is attacked by any of your pieces
        if board.is_attacked_by(color, square):
            # Convert the square (0 to 63) to 8x8 coordinates (rank, file)
            row, col = divmod(square, 8)
            # Mark as attacked
            attacked_map[row, col] = 1
    
    return attacked_map


# ----------------------------------------------- These are funcitons for models with output size: 64 and 6... --------------------------------------
 

def encode_output_piece(move,y,prob):

    moves_plane = encode_move(move)

    y[moves_plane,[int(move[1])-1],table_dict[move[0]]] += prob

    return y
