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
        for i in range(self.iterations):
            new_vals = util.Counter()
            for s in self.mdp.getStates():
                if self.mdp.isTerminal(s):
                    new_vals[s] = 0
                else:
                    q_values = []
                    for a in self.mdp.getPossibleActions(s):
                        q = 0
                        for next_s, p in self.mdp.getTransitionStatesAndProbs(s, a):
                            r = self.mdp.getReward(s, a, next_s)
                            q += p * (r + self.discount * self.values[next_s])
                        q_values.append(q)
                    new_vals[s] = max(q_values) if q_values else 0
            self.values = new_vals

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
        Q = 0
        for next_state, p in self.mdp.getTransitionStatesAndProbs(state, action):
            r = self.mdp.getReward(state,action, next_state)
            Q += p * (r + self.discount * self.values[next_state])
        return Q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        res = None
        max_sofar = float('-inf')
        for a in actions:
            new_Q = self.computeQValueFromValues(state, a)
            if new_Q > max_sofar:
                max_sofar = new_Q
                res = a

        return res

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
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
        predecessors = {s: set() for s in self.mdp.getStates()}
        for s in self.mdp.getStates():
            for a in self.mdp.getPossibleActions(s):
                for (s_next, p) in self.mdp.getTransitionStatesAndProbs(s, a):
                    if p > 0:
                        predecessors[s_next].add(s)
        pq = util.PriorityQueue()
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                max_Q = max(self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s))
                difference = abs(self.values[s] - max_Q)
                pq.update(s, -difference)

        for i in range(self.iterations):
            if pq.isEmpty():
                break
            s = pq.pop()
            if not self.mdp.isTerminal(s):
                self.values[s] = max(self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s))

            for p in predecessors[s]:
                if not self.mdp.isTerminal(p):
                    p_max_Q = max(self.computeQValueFromValues(p, a) for a in self.mdp.getPossibleActions(p))
                    difference = abs(self.values[p] - p_max_Q)
                    if difference > self.theta:
                        pq.update(p, -difference)