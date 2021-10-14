#ML algorithm testing
import pandas as pd
import numpy as np
from sklearn import neural_network
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split


def Classifier():
    #importing the data from the CSV file
    data=pd.read_csv('/Users/benja/Documents/coding practices/X and Y data.csv')
    #this translates the master grid of data into just X moves or just O moves based on the 'oturn' column value, then cuts some of the columns off
    Xcoorddata=(data[(data.oturn==0)]).drop(columns=['winner','oturn','game number'])
    Ocoorddata=(data[(data.oturn==1)]).drop(columns=['winner','oturn','game number'])
    #this is in place to ensure the column headings correspond to the right columns after chopping out some
    Xcoorddata.reset_index(drop=True,inplace=True)
    Ocoorddata.reset_index(drop=True,inplace=True)
    #this extracts the data in the winner column, and separates it into who won in X and O positions
    Xscore=[winner for winner in data[(data.oturn==0)]['winner']]
    Oscore=[winner for winner in data[(data.oturn==1)]['winner']]
    #this section created a test set for testing, ignore it
    #features_train, features_test, labels_train, labels_test = \
    #train_test_split(Ocoorddata, Oscore, test_size=0.1, random_state=42)
    #this section, when utilised, uses a decision tree classifier to fit the data
    '''
    tree=DecisionTreeClassifier(random_state=0,max_leaf_nodes=2000)

    Xmodel=DecisionTreeClassifier(random_state=0,max_leaf_nodes=2000).fit(Xcoorddata,Xscore)
    Omodel=DecisionTreeClassifier(random_state=0,max_leaf_nodes=2000).fit(Ocoorddata,Oscore)
    Xacc=Xmodel.score(Xcoorddata,Xscore)
    Oacc=Omodel.score(Ocoorddata,Oscore)
    print('x tree accuracy=',Xacc)
    print('O tree accuracy=',Oacc)
    
    #this section uses a multi-layer perceptron regression function to fit the data, but since we want a discrete "win/loss" prediction this is not as relevant
    
    from sklearn.neural_network import MLPRegressor
    neural=MLPRegressor(random_state=1, max_iter=3500, solver='lbfgs', early_stopping=True, activation='logistic')
    xneural=neural.fit(Xcoorddata,Xscore)
    oneural=neural.fit(Ocoorddata,Oscore)

    xneuralacc=xneural.score(Xcoorddata,Xscore)
    oneuralacc=oneural.score(Ocoorddata,Oscore)
    print('X neural accuracy=',xneuralacc)
    print('O neural accuracy=',oneuralacc)
    #this section uses a logistic regression to fit the data
    from sklearn.linear_model import LogisticRegression
    LR=LogisticRegression(multi_class='auto', solver='lbfgs')
    xlog=LR.fit(Xcoorddata,Xscore)
    olog=LR.fit(Ocoorddata,Oscore)
    xlogscore=xlog.score(Xcoorddata,Xscore)
    ologscore=olog.score(Ocoorddata,Oscore)
    print('X logistic regression acc=',xlogscore)
    print('o logistic regression acc=',ologscore)
    '''
    
    
    #this classifier uses Multi-layer perceptron to fit the data, and was found to be the most effective
    #NB, perceptron is a simplistic neural network function
    from sklearn.neural_network import MLPClassifier
    MLPC=MLPClassifier(random_state=1, max_iter=1000, activation='relu')
    xfullMLPC = MLPClassifier(random_state=1, max_iter=1000, activation='relu').fit(Xcoorddata,Xscore)
    ofullMLPC = MLPClassifier(random_state=1, max_iter=1000, activation='relu').fit(Ocoorddata,Oscore)
    xMLPCacc=xfullMLPC.score(Xcoorddata,Xscore)
    oMLPCacc=ofullMLPC.score(Ocoorddata,Oscore)
    print('MLPC accuracy: for X={},for O={}'.format(xMLPCacc,oMLPCacc))
    #this section created a preliminary test set for the data so i could inspect exactly which move it was selecting,ignore
    #numbers=np.array([[1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0],[1,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0],[1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0],[1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0],[1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0],[1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1]])
    #labels=['XSQ1','XSQ2','XSQ3','XSQ4','XSQ5','XSQ6','XSQ7','XSQ8','XSQ9','OSQ1','OSQ2','OSQ3','OSQ4','OSQ5','OSQ6','OSQ7','OSQ8','OSQ9']
    #testdata=pd.DataFrame(numbers,columns=labels)
    #print(testdata)
    #predmatrix=ofullMLPC.predict_proba(testdata)
    #print(predmatrix[0])
    #print(predmatrix[1])
    #print(predmatrix)
    
    return xfullMLPC,ofullMLPC

#creates a 3x3 blank numpy array to use as the game board
def creategrid():
    grid=np.zeros([3,3])
    
    return grid

#this returns a bool for whether it is X's or O's turn (True means X)
def Xturn(turncount):

    if (turncount % 2) ==0:
        
        return True

    else:
        
        return False

#this takes the move coordinates from either the ML function or from user input, checks they are valid(that box is empty)
#then adds either a 1 or -1 to that box depending on if it was X or O move respectively
def clickbox(grid,count,coords):
    turn=Xturn(count)
    if turn ==True:
        stringcoords=input('please enter move coords as x,y: ')
        stringcoords=stringcoords.split(",")
        coords=[]
        for i in range(len(stringcoords)):
            temp=int(stringcoords[i])
            coords.append(temp)
    else: 
        coords=coords
    
    #print("remember this counts row 0 as the top row!")
    
    # Change the x/y screen coordinates to grid coordinates
    column = int(coords[1])
    row = int(coords[0])
    
    if grid[row][column]!=0:
        print("square already chosen, pick another!")
    # Set that location to one
    elif turn==True:
        grid[row][column]=1  
    else: 
        grid[row][column]=-1
    return grid
#this sums up the rows, columns and diagonals, and declares a victory if any equal 3 or -3
def winnercheck(grid):
    rowtotal=[]
    columntotal=[]
    for i in range(3):
        rowval=np.sum(grid[i])
        colval=np.sum(grid[:,i])
        rowtotal.append(rowval)
        columntotal.append(colval)

    bottomleftdiag=grid[0,0]+grid[1,1]+grid[2,2]
    topleftdiag=grid[2,0]+grid[1,1]+grid[0,2]

    for i in columntotal:
        if i==3:
            print("game over, player 1 wins!!")
            return True
        elif i==-3:
            print("game over, player 2 wins!!")
            return True
    for i in rowtotal:
        if i==3:
            print("game over, player 1 wins!!")
            return True
        elif i==-3:
            print("game over, player 2 wins!!")
            return True
    if bottomleftdiag==3:
        print("game over, player 1 wins!!")
        return True
    elif topleftdiag==3:
        print("game over, player 1 wins!!")
        return True
    elif bottomleftdiag==-3:
        print("game over, player 2 wins!!")
        return True
    elif topleftdiag==-3:
        print("game over, player 2 wins!!")
        return True
    else:
        return False
#this calculates all possible moves that can be made based on the game board, allowing them to 
#be fed to the ML classifier in a way it can understand
def MLmovesmaker(grid):
    boardshapeX,boardshapeO=turndatasaver(grid)
    oindex=[]
    
    for i in range(len(boardshapeX)):
        if boardshapeX[i]==boardshapeO[i]:
           oindex.append(i)
    possibleslist=[]
    for i in range(len(oindex)):
        newO=boardshapeO.copy()
        newO[oindex[i]]=1
        newdata=boardshapeX+newO
        possibleslist.append(newdata)
    
    return possibleslist,oindex
#this function uses the ML classifier and predict_proba to check the likelihood of a win based on 
#likelihood of game being a 0, 1 or 2
#then translates this output into a square for the AI to move 
#changing line 170 to o classifier, changed line 83 to turn =True
def MLcalc(possibleslist,ocoords,xfullMLPC,ofullMLPC):
    labels=['XSQ1','XSQ2','XSQ3','XSQ4','XSQ5','XSQ6','XSQ7','XSQ8','XSQ9','OSQ1','OSQ2','OSQ3','OSQ4','OSQ5','OSQ6','OSQ7','OSQ8','OSQ9']
    data=possibleslist
    testdata=pd.DataFrame(data,columns=labels)
    probmatrix=ofullMLPC.predict_proba(testdata)
    sortedmatrix=sorted(probmatrix,key=lambda x: x[2])
    flatval=np.where(probmatrix==sortedmatrix[-1])
    flatval=flatval[0]
    flatloc=ocoords[flatval[0]]
    fakegrid=np.zeros([1,9])
    fakegrid[0,(flatloc)]=1
    fakegrid=fakegrid.reshape((3,3))
    coords=np.where(fakegrid==1)
    return coords
#this saves the board for each turn, used in the data generator function
def turndatasaver(grid):
    newgrid=np.reshape(grid,9)
    Xdata=np.where(newgrid==-1,0,newgrid)
    Odata=np.where(newgrid==1,0,newgrid)
    Odata=np.where(Odata==-1,1,Odata)
    Xdata=Xdata.tolist()
    Odata=Odata.tolist()
    return Xdata,Odata
#main function, gonna add a GUI and a randomiser so you either play vs X or vs O
#might also turn this into a generator too so i have a much better dataset, to see if i can improve the 
#classifier accuracy scores, which are currently ~0.83
def main():
    xfullMLPC,ofullMLPC=Classifier()
    finished=False
    while not finished:

        done =False
        count=0
        grid=creategrid()
        while not done:
            possibleslist,ocoords=MLmovesmaker(grid)
            coords=MLcalc(possibleslist,ocoords,xfullMLPC,ofullMLPC)
            print(grid)
            print('\n')
            grid=clickbox(grid,count,coords)
            done=winnercheck(grid)
            count=count+1
            if count>=9:
                done=True
                print('game is a draw')
        replay=input("wanna play again?(y/n):")
        if replay=='y':
            finished=False
        else: 
            finished=True

if __name__ == "__main__":
    main()


