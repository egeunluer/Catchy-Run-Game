#  Opponent Variables Clarification

_opponent_policy and _opponent_pool are config slots. _current_opponent is the active instance for the current episode. They serve different purposes and exist at different layers.
                                                                                                                                                                            
  What each one stores

  self._opponent_policy: Callable | None    # ONE opponent function. Single source.                                                                                         
  self._opponent_pool:   list[Callable] | None  # MANY opponent functions. Pool source.                                                                                     
  self._current_opponent: Callable          # The one chosen for THIS episode.                                                                                              
                                                                                                                                                                            
  When each gets set
                                                                                                                                                                            
  ## CONFIG: pick the source (called from outside the env, by training setup or a callback)
  env.set_opponent(some_callable)               # → sets _opponent_policy                                                                                                   
  env.set_opponent_pool([cb1, cb2, cb3])        # → sets _opponent_pool                                                                                                     
                                                                                                                                                                            
  ## RUNTIME: inside reset(), the env picks who to play this episode                                                                                                         
  def reset(...):                                                                                                                                                           
      self._current_opponent = self._sample_opponent()   # ← _current_opponent gets assigned here 
  
So during a step, only _current_opponent is touched. The other two are only consulted at episode boundaries.  

The first two answer "where do opponents come from"; the third answers "who am I playing right now." They live at different layers because they answer different questions. 