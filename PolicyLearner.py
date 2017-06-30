import tensorflow as tf
import tensorflow.contrib.layers as tfLayers
import numpy as np

class PolicyLearner:
    
    def initVariables(self):
        self.StateHistory = []
        self.RewardHistory = []
        self.ActionHistory = []
        self.Weights = []
        self.Biases = []
        self.WeightHistory = []
        return
        
    def defineLayers(self, layerDims, activation):
        self.X = tf.placeholder(tf.float32, shape = [None,layerDims[0]])
        layer = self.X
        for l in layerDims[1:]:
            layer, w, b = self.neuron_layer(layer, l, activation = activation)
            self.Weights.append(w)
            self.Biases.append(b)
        self.Y = layer
        return
    
    def __init__(self, layerDims, activation = None, 
                 learningRate = 0.1, discountRate = 0.95, regularizationRatio = 1.0,
                logUpdates = False):
        #Set a bunch of stuff to []
        self.initVariables()
        self.DiscountRate = discountRate
        self.LogUpdates = logUpdates
        
        #Initialize the FFNN connections
        self.defineLayers(layerDims, activation)
        
        #Generate the action by sampling from the probability dist in self.Y
        #Turn that into a one-hot label
        self.Action = tf.multinomial(tf.log(tf.nn.softmax(self.Y)),1)
        self.Labels = tf.reduce_mean(
            tf.one_hot(self.Action, layerDims[-1],on_value = 1.0, off_value = 0.0, axis = 1),
            axis = 2)
        
        #Set up the optimizer/gadient calculations
        self.defineLearning(learningRate,regularizationRatio)
        return
    
    def defineLearning(self,learningRate,regRatio):
        self.Loss = self.defineLossFunction(regRatio)
        
        #Choose the optimizer, and define the gradients
        optimizer = tf.train.GradientDescentOptimizer(learningRate)
        gradsAndVars = [(grad, var) for grad, var in optimizer.compute_gradients(self.Loss)
                           if grad != None]
        
        #swap out the gradients for placeholders, so we can do some math on the 
        #grad evaluation later before we run the training operation
        self.GradPlaceholders = []
        self.Gradients = []
        gradVarMap = []
        for grad, var in gradsAndVars:
            self.Gradients.append(grad)
            g = tf.placeholder(tf.float32, shape = grad.get_shape())
            self.GradPlaceholders.append(g)
            gradVarMap.append((g, var))
        
        #Define the training operation
        self.Trainer = optimizer.apply_gradients(gradVarMap)
        return
    
    def defineLossFunction(self, regRatio):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.Labels, logits = self.Y)
        #Idk. this regularization 100% does not work.
        #loss = tfLayers.apply_regularization(tfLayers.l2_regularizer(regRatio),self.Weights)
        return loss
    
    def recordGame(self, stateHistory, actionHistory, rewardHistory):
        self.StateHistory.append(stateHistory)
        self.RewardHistory.append(self.processRewards(rewardHistory))
        self.ActionHistory.append(actionHistory)
        return
    
    def applyUpdate(self):
        #This happened, so heres a check.
        if(len(self.RewardHistory)) == 0: 
            print('nothin!') 
            return
        
        normedRewards = self.normalizeRewards(self.RewardHistory)
        allGrads = self.calcGradients()
        feed_dict = {}
        for nGrad, placeholder in enumerate(self.GradPlaceholders):
            gradMean = np.sum([reward * (allGrads[nGame][nState][nGrad]) 
                                for nGame, rewards in enumerate(normedRewards) 
                                for nState, reward in enumerate(rewards)],
                              axis = 0)
            feed_dict[placeholder] = -gradMean #KEVIN. HERE IS THE - SIGN.
        if self.LogUpdates:
            self.WeightHistory.append([w.eval() for w in self.Weights])
        self.Trainer.run(feed_dict = feed_dict)
        self.StateHistory = []
        self.RewardHistory = []
        self.ActionHistory = []
        return
    
    def calcGradients(self):
        allGrads = []
        states = self.StateHistory
        acts = self.ActionHistory
        for i in range(len(states)):
            oneGameGrads = []
            for j in range(len(states[i])):
                oneGameGrads.append([grad.eval(feed_dict = {self.X : states[i][j], self.Labels : acts[i][j]}) 
                                     for grad in self.Gradients])
            allGrads.append(oneGameGrads)
        return allGrads
    
    def normalizeRewards(self,rewards):
        flat = np.concatenate(rewards)
        mean = np.mean(flat)
        std = np.std(flat)
        normalized = [(r - mean)/std for r in rewards] 
        return normalized
    
    def processRewards(self, rewardHistory):
        processed = np.empty(len(rewardHistory))
        propogatingReward = 0
        for i in reversed(range(len(rewardHistory))):
            propogatingReward = rewardHistory[i] + propogatingReward * self.DiscountRate
            processed[i] = propogatingReward
        return processed

    def neuron_layer(self, X, n_neurons, name = 'layer', activation = None):
        with tf.name_scope(name):
            n_inputs = int(X.get_shape()[1])
            stddev = 2 / np.sqrt(n_inputs)
            init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
            W = tf.Variable(init, name="weights")
            b = tf.Variable(tf.zeros([n_neurons]), name="bias")
            Z = tf.matmul(X, W) + b
            if activation is not None:
                Z = activation(Z)
            return Z, W, b 
    
    def getAction(self, state):
        return self.Labels.eval(feed_dict = {self.X : state})
    def getLogits(self, state):
        return tf.nn.softmax(self.Y).eval(feed_dict = {self.X : state})

def rms(mat):
    return np.sqrt(np.mean(np.square(mat)))

def testLearner(learner, g):
    nRuns = 200
    nWin = 0
    nLose = 0
    for i in range(nRuns):
        state = g.newGame()
        end = 0
        for nState in range(100):
            action = learner.getAction(state)
            state, end = g.getNextState(state, action)
            if int(end) != 0:
                break
        if end > 0:
            nWin += 1
        elif end < 0:
            nLose += 1
    return nWin/nRuns, nLose/nRuns, (nRuns-nWin-nLose)/nRuns

def newSession():
    sessConfig = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
    return tf.Session(config = sessConfig)

def playGame(g, learner,maxMoves):
    state = g.newGame()
    for nState in range(maxMoves):
        action = learner.getAction(state)
        print(learner.getLogits(state),learner.getAction(state))
        state, end = g.getNextState(state, action)
        rewardHist.append(end)
        if int(end) != 0:
            ended = True
            break
    return ended

from b02f4d5609550a0a04878dc5a54f9c2b import game
g = game()
state = g.newGame()
learner = PolicyLearner(
    [state.shape[1], 10, 10, g.getPossibleMoves()],
    learningRate = 0.11,
    discountRate = 0.8,
    regularizationRatio = 0.00001, # this doesn't work
    activation = tf.nn.relu,
    logUpdates = True)

nIterations = 10
nGames = 50
maxMoves = 2000

with newSession():
    tf.global_variables_initializer().run()
    for i in range(nIterations):
        for nGame in range(nGames):
            state = g.newGame()
            stateHist = []
            actionHist = []
            rewardHist = []
            ended = False
            for nState in range(maxMoves):
                stateHist.append(state)
                action = learner.getAction(state)
                actionHist.append(action)
                state, end = g.getNextState(state, action)
                rewardHist.append(end)
                if int(end) != 0:
                    ended = True
                    break
            if(ended): 
                learner.recordGame(stateHist,actionHist,rewardHist)
        learner.applyUpdate()
        print('Win Rate: {0}'.format(testLearner(learner,g)))
        print('Weight magnitude',[rms(w) for w in learner.WeightHistory[-1]])
        #playGame(g,learner,maxMoves)

