#imports
import numpy as np #supporting library sometimes used with pandas, just a nice dependency to account for
import pandas as pd #handling df
import math #complex functions made easy


#data denotion for map operations
class TMat():
  def __init__(self):
    # two maps of mats, one is for counting and the other is to represent probabilities
    self.map = {
      "A": {"A": 0, "T": 0, "C": 0, "G": 0},
      "T": {"A": 0, "T": 0, "C": 0, "G": 0},
      "C": {"A": 0, "T": 0, "C": 0, "G": 0},
      "G": {"A": 0, "T": 0, "C": 0, "G": 0}
    }

  def train(self, df):  # pass in a dataframe
    x, y = df.shape  # x is rows per linear algebra
    for row in range(x):
      seq = df.iloc[row, 1]  # get the RNA sequence
      for let in range(len(seq[0:-1])):  # don't count last entry, otherwise bound error
        self.map[seq[let]][seq[let + 1]] += 1  # adjust the counter map based on transition instance

  def normalize(self, df):  # adjust the data into percentages
    x, y = df.shape
    norm_map = { #norm map used to pass in to instantiate TNMat
      "A": {"A": 0, "T": 0, "C": 0, "G": 0},
      "T": {"A": 0, "T": 0, "C": 0, "G": 0},
      "C": {"A": 0, "T": 0, "C": 0, "G": 0},
      "G": {"A": 0, "T": 0, "C": 0, "G": 0}
    }
    trans_dict = {"A": 0, "T": 0, "C": 0, "G": 0}  # create map to get divisors
    for chg in trans_dict:  # go through for each letter
      for row in range(x):  # go through all seqs
        seq = df.iloc[row, 1][0:-1]  # last letter is not a transition but an end, no counting after
        trans_dict[chg] += seq.count(chg)  # count number of transitions

    for base in self.map:  # for each of the enrtries in the counter map
      for base2 in self.map:  # for each entry in the entry of counter map (map of maps)
        replace = self.map[base][base2] / trans_dict[
          base]  # divide the number of transition instances with total number of base pair instances
        norm_map[base][base2] = replace  # add that to normalized matrix
    new_obj = TNMat(norm_map)
    return new_obj


class TNMat():
  def __init__(self, norm_map):
    self.map = norm_map

  def testing(self, df, mat):
    accuracy = 0  # initialize the accuracy
    # number of transitions is n-1
    # to get the equilibrium value we can raise .25 to the n-1
    x, y = df.shape
    for row in range(x):
      seq = df.iloc[row, 1]
      prob = self.map[seq[0]][seq[1]]  # the probability is initially set as the probability of the first transition
      for char in range(len(seq[1:-1])):  # for all other transition length
        prob *= self.map[seq[1:][char]][seq[1:][char + 1]]  # multiply probability iteratively by the new transition probability
      equil_prob = math.pow(.25, len(seq) - 1)  # get the equilibrium probability
      log_ratio = math.log10(float(prob) / float(equil_prob))  # calculate the log ratio of both values
      if log_ratio > 0:  # simplify a 3' output
        log_ratio = 1
      else:  # simplify a 5' output
        log_ratio = 0
      if log_ratio == df.iloc[row, 2]:  # check actual value
        accuracy += 1  # if correct then add one to the numerator of accuracy

    #print(accuracy, x)
    accuracy /= x  # divide the number of correct predictions by number of entries (rows)
    return accuracy  # return the ratio

  def timeSeries(self, seq):
    prob = self.map[seq[0]][seq[1]]
    times = [math.log10(prob/.25)]
    for char in range(len(seq[1:-1])):  # for all other transition length
      prob *= self.map[seq[1:][char]][seq[1:][char + 1]]  # multiply probability iteratively by the new transition probability
      equil_prob = math.pow(.25, len(times) + 1)  # get the equilibrium probability
      log_ratio = math.log10(float(prob) / float(equil_prob)) #caclulate the log ratio
      times.append(log_ratio)

    return times #return the log ratio over the number of transitions

  def predict(self, seq):
    prob = self.map[seq[0]][seq[1]]
    for char in range(len(seq[1:-1])):  # for all other transition length
      prob *= self.map[seq[1:][char]][
        seq[1:][char + 1]]  # multiply probability iteratively by the new transition probability
    equil_prob = math.pow(.25, len(seq) - 1)  # get the equilibrium probability
    log_ratio = math.log10(float(prob) / float(equil_prob))  #caclulate the log ratio

    if log_ratio > 0: #positive represents the value type we trained our initial matrix on
      pred = True
    if log_ratio < 0:
      pred = False #represents another type compared to what structure we trained the matrix on
    return log_ratio, pred

  def predictKnown(self, seq, predict):
    if predict == False: #simplify false to 0
      predict = 0
    elif predict == True: #simplify true to 1
      predict = 1
    prob = self.map[seq[0]][seq[1]]
    for char in range(len(seq[1:-1])):  # for all other transition length
      prob *= self.map[seq[1:][char]][seq[1:][char + 1]]  # multiply probability iteratively by the new transition probability
    equil_prob = math.pow(.25, len(seq) - 1)  # get the equilibrium probability
    log_ratio = math.log10(float(prob) / float(equil_prob)) #caclulate the log ratio
    if log_ratio > 0:  # simplify a 3' output
      num_pred = 1
    else:  # simplify a 5' output
      num_pred = 0
    if num_pred == predict: #prediction is what we expected
      return log_ratio, True
    else:
      return log_ratio, False #prediction is incorrect given the state



