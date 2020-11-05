# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        iterations = self.iterations
        for i in range(iterations):
            mdp = self.mdp
            values = self.values
            states = self.mdp.getStates()
            discount = self.discount

            newvalues = self.values.copy()
            for state in states:
                if mdp.isTerminal(state) is False:
                    actions = mdp.getPossibleActions(state)
                    nextv = float("-inf")
                    for action in actions:
                        q = self.computeQValueFromValues(state, action)
                        if q > nextv:
                            nextv = q
                    newvalues[state] = nextv
            self.values = newvalues
                    
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        values = self.values
        transitions = mdp.getTransitionStatesAndProbs(state, action)
        discount = self.discount
        q = 0
        for transition in transitions:
            reward = mdp.getReward(state, action, transition[0])
            q += transition[1] * (reward + discount * values[transition[0]])
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        actions = mdp.getPossibleActions(state)

        qmax = float("-inf")
        bestAction = None
        for action in actions:
            q = self.computeQValueFromValues(state, action)
            if q > qmax:
                qmax = q
                bestAction = action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
  
    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        iterations = self.iterations
        discount = self.discount

        i = 0
        while i < iterations:
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state) is False:
                    actions = self.mdp.getPossibleActions(state)
                    nextv = float("-inf")
                    for action in actions:
                        q = self.computeQValueFromValues(state, action)
                        if q > nextv:
                            nextv = q
                    self.values[state] = nextv
                i += 1
                if i >= iterations:
                    return

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        states = mdp.getStates()

        totalPreds = {}
        for state in states:
            totalPreds[state] = self.predecessors(state)
            
        queue = util.PriorityQueue()

        for state in states:
            if mdp.isTerminal(state) is False:
                actions = self.mdp.getPossibleActions(state)
                nextv = float("-inf")
                for action in actions:
                    q = self.computeQValueFromValues(state, action)
                    if q > nextv:
                        nextv = q
                diff = abs(self.values[state] - nextv)
                queue.push(state, -diff)

        for i in range(self.iterations):
            if queue.isEmpty():
                return
            state = queue.pop()
            if mdp.isTerminal(state) is False:
                self.values[state] = max([self.computeQValueFromValues(state, action) for action in mdp.getPossibleActions(state)])
            for pred in totalPreds[state]:
                actions = self.mdp.getPossibleActions(pred)
                nextv = float("-inf")
                for action in actions:
                    q = self.computeQValueFromValues(pred, action)
                    if q > nextv:
                        nextv = q
                diff = abs(self.values[pred] - nextv)
                if diff > self.theta:
                    queue.update(pred, -diff)

 
        
    def predecessors(self, state):
        mdp = self.mdp
        states = mdp.getStates()

        predecessors = []
        for s in states:
            for action in mdp.getPossibleActions(s):
                for transition in mdp.getTransitionStatesAndProbs(s, action):
                    if transition[0] == state and s not in predecessors:
                        predecessors.append(s)
        return predecessors


