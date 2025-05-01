import io

import numpy as np
import socket
import struct
import time
import zlib


def send_arrays(sock, arrays, names=None, compress=True):
    # Create a BytesIO object to hold the serialized data
    buffer = io.BytesIO()

    # Save arrays to the buffer
    if names:
        np.savez(buffer, **dict(zip(names, arrays)))
    else:
        np.savez(buffer, *arrays)

    # Get the buffer contents
    data = buffer.getvalue()

    # Send the size of the data first
    sock.sendall(struct.pack('!I', len(data)))

    # Then send the data itself
    sock.sendall(data)


def recv_arrays(sock, decompress=True):
    # Receive the size of the incoming data
    size = struct.unpack('!I', sock.recv(4))[0]

    # Receive the data
    data = b''
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet

    # Load the arrays from the received data
    return dict(np.load(io.BytesIO(data)))


def desktop_send(sock, arr):
    assert arr.shape == (3,), "Array shape must be (3,)"
    send_arrays(sock, [arr], names=["control"])


def spot_send(sock, rgb_image, depth_image, transform_matrix):
    assert rgb_image.ndim == 3 and rgb_image.shape[2] == 3, "RGB image must be 3D with 3 channels"
    assert depth_image.ndim == 2, "Depth image must be 2D"
    # assert transform_matrix.shape == (3, 3), "Transform matrix must be 3x3"

    send_arrays(sock, [rgb_image, depth_image, transform_matrix], names=["rgb", "depth", "tf"])


def desktop_recv(sock):
    rcv = recv_arrays(sock)
    return rcv


def spot_recv(sock):
    rcv = recv_arrays(sock)
    return rcv


def setup_server(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', port))
    s.listen()
    conn, addr = s.accept()
    return conn


def setup_client(port, max_retries=40, retry_delay=1):
    retries = 0
    while retries < max_retries:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(('localhost', port))
            return s
        except ConnectionRefusedError:
            print(f"Connection failed. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retries += 1

    raise ConnectionError("Maximum retries reached. Unable to connect to the server.")

# Test script for Side 1
def test_side1():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 12345))
        s.listen()
        print("Side 1: Waiting for connection...")
        conn, addr = s.accept()
        with conn:
            print(f"Side 1: Connected by {addr}")

            # Send data
            a = time.time()
            for i in range(100):
                send_data = np.array([1.0, 2.0], dtype=np.float32)
                print(f"Side 1: Sending data: {send_data}")
                desktop_send(conn, send_data)

                # Receive data
                rgb, depth, transform = desktop_recv(conn)
                print("Side 1: Received data:")
                print(f"sums: {rgb.sum(), depth.sum(), transform.sum()}")

            print((time.time() - a) / 100.0)

# Test script for Side 2
def test_side2():
    time.sleep(1)  # Give Side 1 time to start
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 12345))
        print("Side 2: Connected to server")

        # Receive data
        received_data = spot_recv(s)
        print(f"Side 2: Received data: {received_data}")

        # Send data
        rgb_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        depth_image = np.random.rand(480, 640).astype(np.float32)
        transform_matrix = np.random.rand(3, 3).astype(np.float32)

        print("Side 2: Sending data:")
        print(f"RGB image shape: {rgb_image.shape}")
        print(f"Depth image shape: {depth_image.shape}")
        print(f"Transform matrix:\n{transform_matrix}")

        spot_send(s, rgb_image, depth_image, transform_matrix)

        received_data = spot_recv(s)
        print(f"Side 2: Received data: {received_data}")

        # Send data
        rgb_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        depth_image = np.random.rand(480, 640).astype(np.float32)
        transform_matrix = np.random.rand(3, 3).astype(np.float32)

        print("Side 2: Sending data:")
        print(f"RGB image shape: {rgb_image.shape}")
        print(f"Depth image shape: {depth_image.shape}")
        print(f"Transform matrix:\n{transform_matrix}")

        spot_send(s, rgb_image, depth_image, transform_matrix)


if __name__ == "__main__":
    import multiprocessing

    p1 = multiprocessing.Process(target=test_side1)
    # p2 = multiprocessing.Process(target=test_side2)
    p1.start()
    # p2.start()
    p1.join()
    # p2.join()
