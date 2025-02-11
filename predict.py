from auxiliary_functions import num2files,create_position_planes,table_dict
import numpy as np
import torch
import torch.nn.functional as F

    
def decode_plane(plane, f1,r1):

    # Initialize "to" move
    f2, r2 = None, None

    if 0 <= plane <= 7:
        # Knight moves
        if plane == 0:
            r2 = r1 + 2
            f2 = f1 - 1
        elif plane == 1:
            r2 = r1 + 2
            f2 = f1 + 1
        elif plane == 2: # perhaps...
            r2 = r1 - 2
            f2 = f1 - 1
        elif plane == 3:
            r2 = r1 - 2
            f2 = f1 + 1
        elif plane == 4:
            r2 = r1 + 1
            f2 = f1 - 2
        elif plane == 5:
            r2 = r1 + 1
            f2 = f1 + 2
        elif plane == 6:
            r2 = r1 - 1
            f2 = f1 - 2
        elif plane == 7:
            r2 = r1 - 1
            f2 = f1 + 2


        '''
        if plane in [0, 2, 4, 6]:
            f2 = f1 - 1
        else:
            f2 = f1 + 1

        if plane in [0, 1, 4, 5]:
            r2 = r1 + 2
        else:
            r2 = r1 - 2

        if plane in [2, 3, 6, 7]:
            r2 = r1 - 1 if r2 == r1 + 2 else r1 + 1 
        '''

    elif 8 <= plane <= 20:
        # Vertical moves
        diff_r = (plane - 7) if plane <= 13 else (plane - 13)
        if plane <= 13:
            r2 = r1 + diff_r
        else:
            r2 = r1 - diff_r
        f2 = f1  # File stays the same

    elif 21 <= plane <= 34:
        # Horizontal moves
        diff_f = (plane - 21) if plane <= 27 else (plane - 27)
        if plane <= 27:
            f2 = f1 + diff_f
        else:
            f2 = f1 - diff_f
        r2 = r1  # Rank stays the same

    elif 35 <= plane <= 63:
        # Diagonal moves
        diff_r = abs(plane - 35) % 7
        if 35 <= plane <= 41:
            f2 = f1 + diff_r
            r2 = r1 + diff_r
        elif 42 <= plane <= 48:
            f2 = f1 + diff_r
            r2 = r1 - diff_r
        elif 49 <= plane <= 55:
            f2 = f1 - diff_r
            r2 = r1 - diff_r
        else:  # Planes 56-63
            f2 = f1 - diff_r
            r2 = r1 + diff_r

    to_move = num2files[f2] + str(r2+1) # Is this +1 necessary????????????????????????

    return to_move

def decode_move(prediction,moves_to_consider = 4):
    top_values, top_indices = torch.topk(prediction, moves_to_consider)
    top_values = np.array(top_values)[0]
    top_indices = np.array(top_indices)[0]

    list_of_moves = []

    for idx in top_indices:
        plane =  idx // 64
        temp = idx - plane*64
        row = temp // 8  
        print(f'This is row: {row}')
        files = temp - row * 8
        print(f'This is files: {files}')
        print(f'This is plane: {plane}')
        from_move = num2files[files]+str(row+1) # +1??????????????
        to_move = decode_plane(plane,files,row)
        list_of_moves.append(from_move+to_move)

    
    top_values = F.softmax(torch.tensor(top_values), dim=0)
    top_values = top_values.cpu().numpy()

    return list_of_moves,top_values

def predict(model,device,fen):
    position_planes = create_position_planes(fen)
    
    model.eval()
    single_input = torch.tensor([position_planes], dtype=torch.float32)
    
    if torch.cuda.is_available():
        single_input = single_input.to(device)

    with torch.no_grad():
        output = model(single_input)
        
    recommended_moves, recommended_probs = decode_move(output)
    print(f'recommended moves here: {recommended_moves}')

    return recommended_moves , recommended_probs

def turn_back(move):
    new_best_move = []
    new_best_move.append(move[0])
    new_best_move.append(abs(int(move[1])-9)) # Taking one more to shift it to [0...7]
    new_best_move.append(move[2])
    new_best_move.append(abs(int(move[3])-9)) # Taking one more to shift it to [0...7]
    return ''.join(map(str, new_best_move))