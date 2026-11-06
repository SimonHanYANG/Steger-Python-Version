"""Small blocking process transport for one serialized ROI request at a time.

Tasks are sent straight to persistent children. There is no executor feeder
thread or per-task Future. Parent owns workspace lifetime; shutdown joins every
child before that workspace can be reused or removed.
"""
import multiprocessing as mp
from multiprocessing.connection import wait
import traceback


def _serve(connection, initializer, initargs):
    try:
        try:
            initializer(*initargs)
        except BaseException:
            connection.send((False, traceback.format_exc()))
            return
        connection.send((True, None))
        while True:
            item = connection.recv()
            if item is None:
                break
            function, argument = item
            try:
                connection.send((True, function(argument)))
            except BaseException:
                connection.send((False, traceback.format_exc()))
                break
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class DirectPool:
    def __init__(self, workers, initializer, initargs):
        self.workers = workers
        self.initializer = initializer
        self.initargs = initargs
        self.processes = []
        self.connections = []
        self.closed = False

    def _start(self):
        if self.closed:
            raise RuntimeError('DirectPool is closed')
        if self.processes:
            return
        context = mp.get_context('spawn')
        try:
            for _ in range(self.workers):
                parent, child = context.Pipe()
                process = context.Process(target=_serve, args=(child, self.initializer, self.initargs))
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            for connection in self.connections:
                ok, message = connection.recv()
                if not ok:
                    raise RuntimeError(message)
        except BaseException:
            self.shutdown()
            raise

    def map(self, function, arguments):
        self._start()
        tasks = list(arguments)
        output = [None] * len(tasks)
        active = {}
        next_index = 0
        try:
            for connection in self.connections:
                if next_index >= len(tasks):
                    break
                connection.send((function, tasks[next_index]))
                active[connection] = next_index
                next_index += 1
            while active:
                for connection in wait(list(active)):
                    index = active.pop(connection)
                    ok, result = connection.recv()
                    if not ok:
                        raise RuntimeError('Scale process failed:\n' + result)
                    output[index] = result
                    if next_index < len(tasks):
                        connection.send((function, tasks[next_index]))
                        active[connection] = next_index
                        next_index += 1
            return output
        except BaseException:
            self.shutdown()
            raise

    def shutdown(self, wait=True, cancel_futures=True):
        if self.closed:
            return
        self.closed = True
        # Failure during dispatch can leave a child blocked sending a result.
        # Terminate active children before join; graceful close is only needed
        # for an idle pool. All workspaces are owned by the parent.
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join()
            process.close()
        for connection in self.connections:
            connection.close()
