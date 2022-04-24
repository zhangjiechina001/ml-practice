from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
room_ix,bedrooms_ix,population_ix,households_ix=3,4,5,6

class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True):
        self.add_bedrooms_per_room=add_bedrooms_per_room

    def fit(self,X,y=None):
        return  self

    def transform(self,X):
        rooms_per_household=X[:,room_ix]/X[:,households_ix]
        population_per_household=X[:,population_ix]/X[:,households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_rooms=X[:,bedrooms_ix]/X[:,room_ix]
            return  np.c_[X,rooms_per_household,population_per_household,bedrooms_per_rooms]
        else:
            return  np.c_[X,rooms_per_household,population_per_household]