import tkinter as tk
from PIL import Image, ImageTk
from tkinter import PhotoImage
from tkinter import ttk
import sys
import chess
import random
import torch
from model import ChessModel


from predict import predict as predict_next_move
from auxiliary_functions import num2files

class App(tk.Tk):
    def __init__(self, FEN="", style = [],model=None):
        super().__init__()

        # Here I add the model and device for predictions:
        if model:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                self.model = model.to(self.device)
            else:
                self.model = model
        
        # State:
        if not FEN == "":
            self.starting_position = chess.Board(FEN)
        else:
            self.starting_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
            
        self.board = [[0 for col in range(8)] for row in range(8)]
        
        # 1 Here I intilaize the game:
        self.game = chess.Board(self.starting_position)

        self.move_x = 0
        self.move_y = 0

        
        #These will store the complexity values:
        self.cb = 0
        self.cw = 0

        if not style:
            self.white_color ="#FFFF00"
            self.black_color = "#90EE90"
        else:
            self.white_color = style[0]
            self.black_color = style[1]


        self.square_size = 65
        self.board_size = 8

        # Declaration fo pieces:
        self.Q = tk.PhotoImage(file='.\ChessPieces\q_.png')
        self.q = tk.PhotoImage(file='.\ChessPieces\q.png')

        self.K = tk.PhotoImage(file='.\ChessPieces\k_.png')
        self.k = tk.PhotoImage(file='.\ChessPieces\k.png')

        self.R = tk.PhotoImage(file='.\ChessPieces\/r_.png')
        self.r = tk.PhotoImage(file='.\ChessPieces\/r.png')

        self.B = tk.PhotoImage(file='.\ChessPieces\/b_.png')
        self.b = tk.PhotoImage(file='.\ChessPieces\/b.png')

        self.N = tk.PhotoImage(file='.\ChessPieces\/n_.png')
        self.n = tk.PhotoImage(file='.\ChessPieces\/n.png')

        self.P = tk.PhotoImage(file='.\ChessPieces\p_.png')
        self.p = tk.PhotoImage(file='.\ChessPieces\p.png')

        self.ChessPieces = {
            "q": self.q,
            "Q": self.Q,
            "K": self.K,
            "k": self.k,
            "R": self.R,
            "r": self.r,
            "B": self.B,
            "b": self.b,
            "N": self.N,
            "n": self.n,
            "P": self.P,
            "p": self.p
        }

        # Create a Canvas widget
        canvas = tk.Canvas(self, width = self.square_size * self.board_size, height=self.square_size * self.board_size)
        canvas.pack()

        # Draw the chessboard
        for row in range(self.board_size):
            for col in range(self.board_size):
                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                # Alternate between black and white
                color = self.white_color if (row + col) % 2 == 0 else self.black_color
                canvas.create_rectangle(x1, y1, x2, y2, fill=color,tags='board')

        self.title('Position')
        self.geometry('800x800')

        def Take_input():
            INPUT = inputFEN.get("1.0", "end-1c")
            self.load_FEN(canvas,INPUT)

            inputFEN.delete("1.0","end")
            print(INPUT)

            # setting progress bar values:
            self.evaluate()
        
        inputFEN = tk.Text(self, height = 1,
        width = 50,
        bg = "light yellow")
            
        inputFEN.pack()
        
        def motion(event):

            if self.move_x == 0 and self.move_y == 0:
                self.move_x, self.move_y = event.x, event.y
                self.touch_piece(canvas,[self.move_x, self.move_y])
                self.make_a_move(canvas,self.move_x,self.move_y,self.move_x,self.move_y)
                self.move_x, self.move_y = event.x, event.y
            else:
                x_n, y_n = event.x, event.y
                current_move = self.transform2uci(x_n, y_n)
                if chess.Move.from_uci(current_move) in self.game.legal_moves:

                    print(f'Move played: {current_move}')
                    # Here it makes the move:
                    self.game.push(chess.Move.from_uci(current_move))
                    
                    # Here it makes the move on the board:
                    self.touch_piece(canvas,[self.move_x, self.move_y],[x_n,y_n])
                    self.make_a_move(canvas,self.move_x,self.move_y,x_n,y_n)

                    # After each move we evaluate the position...
                    # setting progress bar values:
                    predict()
                    self.evaluate()
                else:
                    print(f"The move {current_move} is not legal.")
                    self.move_x, self.move_y = 0,0
                
        def load_starting_position():
            INPUT = self.starting_position
            self.load_FEN(canvas,INPUT)

        def predict():
            self.T.delete("1.0","end")
            display_move, display_move_probs = predict_next_move(self.model,self.device,self.game.fen()) 
            display_move_probs = 100 * display_move_probs
            display_move = (
                f"{str(display_move[0])}  -  {display_move_probs[0]:.4f}% \n"
                f"{str(display_move[1])}  -  {display_move_probs[1]:.4f}% \n"
                f"{str(display_move[2])}  -  {display_move_probs[2]:.4f}% \n"
                f"{str(display_move[3])}  -  {display_move_probs[3]:.4f}% \n"
            )

            self.T.insert(tk.END, display_move)

        self.bind('<Button-1>', motion)
        self.bind('<space>', load_starting_position())
        
        B = tk.Button(self, text ="Load FEN", command = lambda:Take_input())
        B.place(x=350,y=600)
        B.pack()

        Move_back = tk.Button(self, text ="  <  ", command = lambda:Take_input())
        Move_forward = tk.Button(self, text ="  >  ", command = lambda:predict())

        Move_back.place(x=365,y=610)
        Move_forward.place(x=405,y=610)
        #Move_back.pack()
        #Move_forward.pack()

        # Here are the key binds:
        self.complexity_white = ttk.Progressbar(orient="vertical",length=100, value=self.cw)
        self.complexity_white.place(x=680, y=420, width=20)

        self.complexity_black = ttk.Progressbar(orient="vertical",length=100, value=self.cb)
        self.complexity_black.place(x=680, y=20, width=20)
        
        self.l = tk.Label(self, text = "Move recommended:")
        self.l.place(x=0,y=530)
        self.T = tk.Text(self, height = 20, width = 20)
        self.T.place(x=0,y=550) 

        self.bind("<Escape>", lambda x: self.destroy())

     # utils defs:   
    def touch_piece(self,canvas,selectedpiece,placedpiece=[]):
        if 'movedpiece' in canvas.gettags('movedpiece'):
            canvas.delete('movedpiece')

        curr_x = selectedpiece[1] // self.square_size
        curr_y = selectedpiece[0] // self.square_size

        if placedpiece:
            next_x = placedpiece[1] // self.square_size
            next_y = placedpiece[0] // self.square_size

        for row in range(self.board_size):
            for col in range(self.board_size):
                
                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                # Alternate between black and white
                if row == curr_x and col == curr_y:
                    color = "#66CD00"
                    canvas.create_rectangle(x1, y1, x2, y2, fill=color,tags='movedpiece')
                elif placedpiece and row == next_x and col == next_y:
                    color = "#66CD00"
                    canvas.create_rectangle(x1, y1, x2, y2, fill=color,tags='movedpiece')
                else:
                    color = self.white_color if (row + col) % 2 == 0 else self.black_color
                    canvas.create_rectangle(x1, y1, x2, y2, fill=color,tags='movedpiece')

    def make_a_move(self,canvas,x1,y1, x2,y2):
        
        curr_x = x1 // self.square_size
        curr_y = y1 // self.square_size
        next_x = x2 // self.square_size
        next_y = y2 // self.square_size
        
        temp = self.board[curr_x][curr_y]
        self.board[curr_x][curr_y] = 0
        self.board[next_x][next_y] = temp

        if 'position' in canvas.gettags('position'):
            canvas.delete('position')
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == 0:
                    continue

                canvas.create_image(row * self.square_size + self.square_size // 2,
                                 col * self.square_size + self.square_size // 2,
                                   image=self.ChessPieces[self.board[row][col]],tag = 'position')
        self.move_x = 0
        self.move_y = 0

    def load_FEN(self,canvas, FEN):
        self.game = chess.Board(FEN)

        if 'position' in canvas.gettags('position'):
            canvas.delete('position')

        self.board = [[0 for col in range(8)] for row in range(8)]
        
        row = 0
        col = 0

        for curr in FEN:
            if curr.isdigit():
                row += int(curr)
                continue

            if curr == '/':
                col += 1
                row = 0
                continue

            if curr == '\n':
                continue

            canvas.create_image(row * self.square_size + self.square_size // 2,
                                 col * self.square_size + self.square_size // 2,
                                   image=self.ChessPieces[curr],tag = 'position')
            
            self.board[row][col] = curr

            row += 1
    
    #transform coordinates to uci:
    def transform2uci(self,x_n, y_n):
        prev_x = self.move_x // self.square_size
        # because 0,0 coords are in the top left,
        # the calculation of squares starts differently
        # that is why I need to adjust:
        prev_y = abs(self.move_y // self.square_size -7)

        curr_x = x_n // self.square_size
        curr_y = abs(y_n // self.square_size-7) 

        move = num2files[prev_x]+str(prev_y+1)+num2files[curr_x]+ str(curr_y+1)
        return move

    # This part evaluates the complexity for both black and white
    # and changes progressbar values accordingly:
    def evaluate(self):
        
        self.complexity_white.config(value=random.randint(0,100))
        self.complexity_black.config(value=random.randint(0,100))

if __name__ == "__main__":

    # These are just different types of chess tables:
    STYLE1 = ["#FEFAE0","#DDA15E"]
    STYLE2 = ["#FDF0D5", "#83C5BE"]
    STYLE3 = ["#D3D3D3", "#5FA8D3"]
    STYLE4 = ["#FFFCF2", "#CB997E"]
    STYLE5 = ["#DDB892", "#9C6644"]
    STYLE6 = ["#C8B8DB", "#70587C"]
    STYLE7 = ["#AA998F", "#7D4F50"]
    STYLE8 = ["#F2E9E4", "#CC8B86"]
    
    model = ChessModel()
    model.load_state_dict(torch.load('pretrained_models/NOTURNING_SIMPLEMODEL_14LAYERINPUT.pth',map_location=torch.device('cpu')),strict=False)

    app = App(style=STYLE1,model=model)
    app.mainloop()

