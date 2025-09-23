#%%
"""
Quant Challenge 2025

Algorithmic strategy template
"""

from enum import Enum
from typing import Optional

class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    # TEAM_A (home team)
    TEAM_A = 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    """Place a market order.
    
    Parameters
    ----------
    side
        Side of order to place
    ticker
        Ticker of order to place
    quantity
        Quantity of order to place
    """
    return

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    """Place a limit order.
    
    Parameters
    ----------
    side
        Side of order to place
    ticker
        Ticker of order to place
    quantity
        Quantity of order to place
    price
        Price of order to place
    ioc
        Immediate or cancel flag (FOK)

    Returns
    -------
    order_id
        Order ID of order placed
    """
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    """Cancel an order.
    
    Parameters
    ----------
    ticker
        Ticker of order to cancel
    order_id
        Order ID of order to cancel

    Returns
    -------
    success
        True if order was cancelled, False otherwise
    """
    return 0

import math
import time
from collections import defaultdict, deque

home_edge = 0.02 #preference for a home team
early_point = 0.05 # value of 4% in the beginning of the game
later_point = 0.10 #value of 15% closer to the end of the game

enter_threshold = 0.05 #enter the contract when at least 5% price difference
exit_threshold = 0.02
max_pos = 5 #max amount of contract allowed to hold at the same time
order_quantity = 1
cooldown_sec = 1.0
orders_per_min = 30
flatten_t_final = 9 #9 seconds efore the game end force flatten
last_coin_flip_sec = 9.0 #in the last 9 seconds win prob is 50/50
use_market_orders = True


class Strategy:
    """Template for a strategy."""

    def _clamp01(x):
        """Does not allows probabilities to be larger than 1 and lower than 0"""
        return 0.0 if x<0 else 1.0 if x>1 else x

    def reset_state(self) -> None:
        """Reset the state of the strategy to the start of game position.
        
        Since the sandbox execution can start mid-game, we recommend creating a
        function which can be called from __init__ and on_game_event_update (END_GAME).

        Note: In production execution, the game will start from the beginning
        and will not be replayed.
        """
        self.bids = defaultdict(float)
        self.asks = defaultdict(float)

        self.best_bid = None
        self.best_asks = None
        self.last_mid = None

        self.position = 0.0
        self.avg_price = 0.0
        self.realized_pnl = 0.0
        self.capital_remaining = None

        self.total_time_guess = 2400.0
        self.last_order_time = 0.0

        self.order_timestamps = deque()
        self.last_order_time = 0.0

        self.last_edge = 0.0

        self.live_orders = {}

    def __init__(self) -> None:
        """Your initialization code goes here."""
        self.reset_state()

    def _now(self):
        return time.monotonic()
    
    def _can_send_orders(self):
        """Respects 30 orders/mins and allows some cooldown"""
        now = self._now()
        self.order_timestamps.append(now)
        self.last_order_time = now

    def _marker_order_sent(self):
        now = self._now()
        self.order_timestamps.append(now)
        self.last_order_time = now

    def _set_book_level(self, side, price, quantity):
        """Maintain local book of prices and quantity"""
        book=self.bids if side == Side.BUY else self.asks
        if quantity<=0:
            if price in book:
                del book[price]
        else:
            book[price] = float(quantity)

    def _update_best_levels(self):
        self.best_bid = max(self.bids.keys()) if self.bids else None
        self.best_asks = min(self.asks.keys()) if self.asks else None
        if self.best_asks is not None and self.best_bid is not None:
            self.last_mid = 0.5*(self.best_bid + self.best_asks)
        elif self.best_bid is not None:
            self.last_mid = self.best_bid
        elif self.best_asks is not None:
            self.last_mid = self.best_asks
        else:
            self.last_mid = None
    
    def _market_prob(self):
        """Converts mid prices to probabilities"""
        if self.last_mid is None:
            return None
        prob = self.last_mid/100.0
        if prob<0.0:
            return 0.0
        if prob>1.0:
            return 1.0
        return prob
    
    def _infer_total_time(self, time_remaining):
        if time_remaining is None:
            return None
        if time_remaining>self.total_time_guess and time_remaining<=2880.0:
            self.total_time_guess=2880.0 if time_remaining>2400.0 else 2400.0

    def _time_progress(self, time_remaining):
        if time_remaining is None:
            return 0.5
        total = max(1.0, self.total_time_guess)
        tr = max(0.0, min(total, time_remaining))
        return (total - tr)/total
    
    def _win_prob_model(self, score_diff, time_remaining):
        """ in the final seconds of th game, prob = 50/50
        else: 0.5+fading home edge+(points x time-weighted value)"""
        if time_remaining is not None and time_remaining<=last_coin_flip_sec:
            return 0.5
        t=self._time_progress(time_remaining)
        fade_home_edge = home_edge*(1.0-t)
        point_val = early_point+(later_point-early_point)*t
        p=0.5+fade_home_edge+score_diff*point_val
        return max(0.0, min(1.0, self.last_mid/100.0))
    
    def _place(self, side, qty):
        """Send an order"""
        if not self._can_send_orders():
            return
        if use_market_orders:
            place_market_order(side, Ticker.TEAM_A, qty)
        else:
            if side==Side.BUY:
                px=self.best_asks if self.best_asks is not None else 100.0
            else:
                px=self.best_bid if self.best_bid is not None else 0.0
            place_limit_order(side, Ticker.TEAM_A, qty, px, ioc=True)
        self._marker_order_sent()

    def on_trade_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        """Called whenever two orders match. Could be one of your orders, or two other people's orders.
        Parameters
        ----------
        ticker
            Ticker of orders that were matched
        side:
            Side of orders that were matched
        quantity
            Volume traded
        price
            Price that trade was executed at
        """
        print(f"Python Trade update: {ticker} {side} {quantity} shares @ {price}")

    def on_orderbook_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        """Called whenever the orderbook changes. This could be because of a trade, or because of a new order, or both.
        Parameters
        ----------
        ticker
            Ticker that has an orderbook update
        side
            Which orderbook was updated
        price
            Price of orderbook that has an update
        quantity
            Volume placed into orderbook
        """
        self._set_book_level(side, price, quantity)
        self._update_best_levels()

    def on_account_update(
        self,
        ticker: Ticker,
        side: Side,
        price: float,
        quantity: float,
        capital_remaining: float,
    ) -> None:
        """Called whenever one of your orders is filled.
        Parameters
        ----------
        ticker
            Ticker of order that was fulfilled
        side
            Side of order that was fulfilled
        price
            Price that order was fulfilled at
        quantity
            Volume of order that was fulfilled
        capital_remaining
            Amount of capital after fulfilling order
        """
        self.capital_remaining=capital_remaining
        qty = float(quantity)
        px = float(price)
        
        if side==Side.BUY:
            new_pos = self.position + qty
            if self.position<0:
                closed=min(qty, -self.position)
                self.realized_pnl+=(self.avg_price-px)*closed
            if new_pos>0:
                long_qty_before = max(0.0, self.position)
                self.avg_price=((self.avg_price*long_qty_before+px*qty)/new_pos)
            else:
                if new_pos==0:
                    self.avg_price=0.0
            self.position = new_pos
        elif side == Side.SELL:
            new_pos=self.position-qty
            if self.position>0:
                closed=min(qty, self.position)
                self.realized_pnl+=(px-self.avg_price)*closed
            if new_pos<0:
                short_qty_before = max(0.0, -self.position)
                self.avg_price=((self.avg_price*short_qty_before+px*qty)/(-new_pos))
            else:
                if new_pos==0:
                    self.avg_price=0.0
            self.position=new_pos
    

    def on_game_event_update(self,
                           event_type: str,
                           home_away: str,
                           home_score: int,
                           away_score: int,
                           player_name: Optional[str],
                           substituted_player_name: Optional[str],
                           shot_type: Optional[str],
                           assist_player: Optional[str],
                           rebound_type: Optional[str],
                           coordinate_x: Optional[float],
                           coordinate_y: Optional[float],
                           time_seconds: Optional[float]
        ) -> None:
        """Called whenever a basketball game event occurs.
        Parameters
        ----------
        event_type
            Type of event that occurred
        home_score
            Home team score after event
        away_score
            Away team score after event
        player_name (Optional)
            Player involved in event
        substituted_player_name (Optional)
            Player being substituted out
        shot_type (Optional)
            Type of shot
        assist_player (Optional)
            Player who made the assist
        rebound_type (Optional)
            Type of rebound
        coordinate_x (Optional)
            X coordinate of shot location in feet
        coordinate_y (Optional)
            Y coordinate of shot location in feet
        time_seconds (Optional)
            Game time remaining in seconds
        """

        print(f"{event_type} {home_score} - {away_score}")

        self.last_time_seen = time_seconds
        self._infer_total_time(time_seconds)

        if event_type == "END_GAME":
            # IMPORTANT: Highly recommended to call reset_state() when the
            # game ends. See reset_state() for more details.
            self.reset_state()
            return
        p_mkt = self._market_prob()
        if p_mkt is None:
            return
        
        score_diff = int(home_score)-int(away_score)
        p_model=self._win_prob_model(score_diff, time_seconds)
        edge = p_model-p_mkt
        self.last_edge=edge

        if (time_seconds is not None) and (time_seconds<=flatten_t_final):
            if self.position>0:
                self._place(Side.SELL, min(order_quantity, abs(self.position)))
            elif self.position<0:
                self._place(Side.BUY, min(order_quantity, abs(self.position)))
            return
        if edge>= enter_threshold and self.position<max_pos:
            self._place(Side.BUY, order_quantity)
            return
        if edge <= -enter_threshold and self.position > - max_pos:
            self._place(Side.SELL, order_quantity)
            return
        
        if abs(edge)<exit_threshold:
            if self.position>0:
                self._place(Side.SELL, order_quantity)
            elif self.position<0:
                self._place(Side.BUY, order_quantity)
            return



    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        """Called periodically with a complete snapshot of the orderbook.

        This provides the full current state of all bids and asks, useful for 
        verification and algorithms that need the complete market picture.

        Parameters
        ----------
        ticker
            Ticker of the orderbook snapshot (Ticker.TEAM_A)
        bids
            List of (price, quantity) tuples for all current bids, sorted by price descending
        asks  
            List of (price, quantity) tuples for all current asks, sorted by price ascending
        """
        # Reset the state of local books
        self.bids.clear()
        self.asks.clear()
        for px, qty in bids:
            if qty>0:
                self.bids[float(px)]=float(qty)
        for px, qty in asks:
            if qty>0:
                self.asks[float(px)]=float(qty)
        self._update_best_levels()
        

    def win_prob_model(self, score_diff, time_remaining):
        if time_remaining is None:
            time_progress = 0.5


#%%
import json

# Load the JSON file with events
with open("example-game.json", "r") as f:
    events = json.load(f)

# Initialize your strategy
strategy = Strategy()

# Replay events into the strategy
for e in events:
    strategy.on_game_event_update(
        event_type=e["event_type"],
        home_away=e["home_away"],
        home_score=e["home_score"],
        away_score=e["away_score"],
        player_name=e.get("player_name"),
        substituted_player_name=e.get("substituted_player_name"),
        shot_type=e.get("shot_type"),
        assist_player=e.get("assist_player"),
        rebound_type=e.get("rebound_type"),
        coordinate_x=e.get("coordinate_x"),
        coordinate_y=e.get("coordinate_y"),
        time_seconds=e.get("time_seconds"),
    )


# %%
