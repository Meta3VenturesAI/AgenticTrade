import websocket
import json
import queue
import threading
import logging
import time
import asyncio
from typing import Any, Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('strategy.log', mode='a', encoding='utf-8')])
logger = logging.getLogger()

class AlphaVantageWS:
    """WebSocket client for streaming Alpha Vantage data.
    
    Args:
        api_key (str): Alpha Vantage API key.
        tickers (List[str]): List of ticker symbols.
        data_queue (queue.Queue): Queue for passing data to the strategy.
    """
    def __init__(self, api_key: str, tickers: List[str], data_queue: queue.Queue):
        self.api_key = api_key
        self.tickers = tickers
        self.data_queue = data_queue
        self.ws = None
        self.running = False
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        """Handle incoming WebSocket messages.
        
        Args:
            ws (websocket.WebSocketApp): WebSocket instance.
            message (str): Received message.
        """
        try:
            with self.lock:
                data = json.loads(message)
                if 'data' in data and any(ticker in data['data'] for ticker in self.tickers):
                    self.data_queue.put(data['data'])
                    logger.info(f"Received data for {data['data'].get('symbol', 'unknown')}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Message handling error: {e}")

    def on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        """Handle WebSocket errors.
        
        Args:
            ws (websocket.WebSocketApp): WebSocket instance.
            error (Exception): Error object.
        """
        logger.error(f"WebSocket error: {error}")
        if not self.stop_event.is_set():
            self.reconnect()

    def on_close(self, ws: websocket.WebSocketApp, close_status_code: int, close_msg: str) -> None:
        """Handle WebSocket closure.
        
        Args:
            ws (websocket.WebSocketApp): WebSocket instance.
            close_status_code (int): Status code.
            close_msg (str): Close message.
        """
        logger.info(f"WebSocket closed with code {close_status_code}: {close_msg}")
        if not self.stop_event.is_set():
            self.reconnect()

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        """Handle WebSocket opening.
        
        Args:
            ws (websocket.WebSocketApp): WebSocket instance.
        """
        logger.info("WebSocket opened")
        subscribe_msg = {"action": "subscribe", "symbols": self.tickers, "apikey": self.api_key}
        ws.send(json.dumps(subscribe_msg))

    def reconnect(self) -> None:
        """Reconnect to WebSocket with exponential backoff."""
        wait_time = 1
        while not self.stop_event.is_set() and wait_time < 60:
            try:
                self.start()
                break
            except Exception as e:
                logger.warning(f"Reconnect attempt failed: {e}. Retrying in {wait_time}s")
                time.sleep(wait_time)
                wait_time *= 2

    def start(self) -> None:
        """Start the WebSocket connection."""
        if not self.running:
            with self.lock:
                if not self.running:
                    self.ws = websocket.WebSocketApp(
                        f"wss://ws.eodhistoricaldata.com/ws/us?api_token={self.api_key}",
                        on_message=self.on_message,
                        on_error=self.on_error,
                        on_close=self.on_close,
                        on_open=self.on_open
                    )
                    self.running = True
                    wst = threading.Thread(target=self.ws.run_forever, daemon=True)
                    wst.start()
                    logger.info("WebSocket thread started")

    def stop(self) -> None:
        """Stop the WebSocket connection."""
        with self.lock:
            if self.running:
                self.stop_event.set()
                if self.ws:
                    self.ws.close()
                self.running = False
                logger.info("WebSocket stopped")

class IBKRTrader:
    """Interactive Brokers trader for executing trades.
    
    Args:
        data_queue (queue.Queue): Queue for receiving data.
        trade_queue (queue.Queue): Queue for sending trade orders.
        dry_run (bool): Whether to execute in dry run mode.
    """
    def __init__(self, data_queue: queue.Queue, trade_queue: queue.Queue, dry_run: bool = False):
        self.data_queue = data_queue
        self.trade_queue = trade_queue
        self.dry_run = dry_run
        self.connected = False
        self.stop_event = asyncio.Event()
        self.order_lock = threading.Lock()
        self.order_status = {}
        self.last_order_id = 0
        self.ib = None  # Defer initialization

    async def connect(self) -> None:
        """Connect to Interactive Brokers TWS or Gateway."""
        try:
            # Import ib_insync only when needed
            import ib_insync
            from ib_insync import util

            self.ib = ib_insync.IB()
            if not self.dry_run:
                # Use util.startLoop() to manage the event loop
                util.startLoop()
                await self.ib.connectAsync("127.0.0.1", 7497, clientId=1)
                self.connected = self.ib.isConnected()
                if self.connected:
                    logger.info("Connected to IBKR")
                    self.ib.pendingTickersEvent += self.on_pending_tickers
                    self.ib.orderStatusEvent += self.on_order_status
                    # Start order monitor in a separate task
                    asyncio.create_task(self.start_order_monitor())
                else:
                    logger.error("Failed to connect to IBKR")
                    self.connected = False
        except Exception as e:
            logger.error(f"IBKR connection error: {e}")
            self.connected = False

    def on_pending_tickers(self, tickers: List[Any]) -> None:
        """Handle pending ticker updates.
        
        Args:
            tickers (List[Any]): List of ticker updates.
        """
        try:
            for ticker in tickers:
                if ticker.contract.symbol in self.data_queue:
                    self.data_queue.put({'symbol': ticker.contract.symbol, 'price': ticker.last})
        except Exception as e:
            logger.error(f"Pending tickers error: {e}")

    def on_order_status(self, trade: Any) -> None:
        """Handle order status updates.
        
        Args:
            trade (Any): Trade object with status.
        """
        with self.order_lock:
            self.order_status[trade.order.orderId] = {
                'filled': trade.orderStatus.filled,
                'status': trade.orderStatus.status,
                'remaining': trade.orderStatus.remaining
            }
            logger.info(f"Order {trade.order.orderId} status: {trade.orderStatus.status}")

    async def start_order_monitor(self) -> None:
        """Monitor and manage order execution."""
        while not self.stop_event.is_set() and self.connected:
            try:
                with self.order_lock:
                    for order_id, status in self.order_status.items():
                        if status['status'] == 'PendingSubmit' and status['remaining'] > 0:
                            self.ib.cancelOrder(order_id)
                            logger.warning(f"Cancelled unfilled order {order_id}")
                            await self.retry_order(order_id)
            except Exception as e:
                logger.error(f"Order monitor error: {e}")
            await asyncio.sleep(1)

    async def retry_order(self, order_id: int, max_retries: int = 3) -> None:
        """Retry a failed or partially filled order.
        
        Args:
            order_id (int): Order ID to retry.
            max_retries (int): Maximum retry attempts.
        """
        with self.order_lock:
            if self.order_status.get(order_id, {}).get('retry_count', 0) < max_retries:
                status = self.order_status[order_id]
                if status['remaining'] > 0:
                    await self.place_order(status['contract'], status['order'], order_id)
                    self.order_status[order_id]['retry_count'] = self.order_status.get(order_id, {}).get('retry_count', 0) + 1
                    logger.info(f"Retrying order {order_id}")

    async def place_order(self, contract: Any, order: Any, order_id: Optional[int] = None) -> None:
        """Place a trade order with IBKR.
        
        Args:
            contract (Any): Contract details.
            order (Any): Order details.
            order_id (Optional[int]): Custom order ID.
        """
        try:
            if not self.dry_run:
                if order_id is None:
                    self.last_order_id += 1
                    order_id = self.last_order_id
                order.orderId = order_id
                await self.ib.placeOrderAsync(contract, order)
                with self.order_lock:
                    self.order_status[order_id] = {'filled': 0, 'status': 'PendingSubmit', 'remaining': order.totalQuantity}
                logger.info(f"Placed order {order_id} for {contract.symbol}")
            else:
                logger.info(f"Dry run: Would place order for {contract.symbol}")
        except Exception as e:
            logger.error(f"Order placement error: {e}")

    async def process_trade(self) -> None:
        """Process trades from the trade queue."""
        while not self.stop_event.is_set() and self.connected:
            try:
                trade = self.trade_queue.get_nowait()
                action = trade.get('action')
                size = trade.get('size', 0.1)
                ticker = trade.get('ticker', 'TQQQ')

                # Import ib_insync classes only when needed
                from ib_insync import Stock, Order

                contract = Stock(ticker, 'SMART', 'USD')
                order = Order()
                order.action = 'BUY' if action == 'BUY' else 'SELL'
                order.totalQuantity = size
                order.orderType = 'MKT'

                await self.place_order(contract, order)
            except queue.Empty:
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Trade processing error: {e}")

    async def disconnect(self) -> None:
        """Disconnect from IBKR and clean up."""
        try:
            if self.connected and not self.dry_run and self.ib:
                await self.ib.disconnectAsync()
                self.connected = False
                logger.info("Disconnected from IBKR")
        except Exception as e:
            logger.error(f"Error disconnecting from IBKR: {e}")
        self.stop_event.set()

    async def run(self) -> None:
        """Run the trader loop."""
        if not self.dry_run:
            await self.connect()
        await self.process_trade()
