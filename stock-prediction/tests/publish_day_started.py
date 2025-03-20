import pika
import json
import uuid
from datetime import datetime
import os
import socket

def publish_day_started():
    # Connection parameters
    host = os.environ.get("RABBITMQ_HOST", "localhost")  # Default to localhost since port is mapped
    port = int(os.environ.get("RABBITMQ_PORT", "5672"))
    username = os.environ.get("RABBITMQ_USERNAME", "guest")
    password = os.environ.get("RABBITMQ_PASSWORD", "guest")

    print(f"Connecting to RabbitMQ at {host}:{port}...")

    # First, test if the host is reachable
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # 5 second timeout
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result != 0:
            raise ConnectionError(f"Could not connect to {host}:{port}. Port is not open.")
    except socket.gaierror:
        raise ConnectionError(f"Could not resolve hostname: {host}")
    except Exception as e:
        raise ConnectionError(f"Connection test failed: {str(e)}")

    # Create connection
    try:
        credentials = pika.PlainCredentials(username, password)
        connection_params = pika.ConnectionParameters(
            host=host,
            port=port,
            credentials=credentials,
            connection_attempts=3,
            retry_delay=2,
            socket_timeout=5,  # 5 second socket timeout
            blocked_connection_timeout=10,  # 10 second blocked connection timeout
            heartbeat=600  # 10 minute heartbeat
        )
        
        print("Attempting to establish connection...")
        connection = pika.BlockingConnection(connection_params)
        channel = connection.channel()
        print("Successfully connected to RabbitMQ")

        # Declare exchange
        channel.exchange_declare(
            exchange='day-started-exchange',
            exchange_type='fanout',
            durable=True
        )
        print("Exchange 'day-started-exchange' declared")

        # Create message
        message_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat(timespec='microseconds') + 'Z'

        message = {
            "messageId": message_id,
            "conversationId": message_id,
            "messageType": ["urn:message:Domain.Time.DomainEvents:DayStarted"],
            "message": {
                "Id": message_id,
                "Timestamp": timestamp
            },
            "sentTime": timestamp,
            "headers": {
                "MT-Activity-Id": message_id,
                "MT-Message-Type": "DayStarted"
            }
        }

        print(f"Publishing message: {json.dumps(message, indent=2)}")

        # Publish message
        channel.basic_publish(
            exchange='day-started-exchange',
            routing_key='',
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
                content_type='application/vnd.masstransit+json',
                content_encoding='utf-8',
                message_id=message_id,
                correlation_id=message_id,
                timestamp=int(datetime.now().timestamp()),
                type='urn:message:Domain.Time.DomainEvents:DayStarted',
                headers={
                    'MT-Activity-Id': message_id,
                    'MT-Message-Type': 'DayStarted'
                }
            )
        )

        print(f"Published DayStarted event with ID: {message_id}")
        connection.close()
        print("Connection closed")
        
    except pika.exceptions.AMQPConnectionError as e:
        raise ConnectionError(f"AMQP Connection Error: {str(e)}")
    except pika.exceptions.AMQPChannelError as e:
        raise ConnectionError(f"AMQP Channel Error: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    try:
        publish_day_started()
    except ConnectionError as e:
        print(f"Connection Error: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Verify RabbitMQ is running: docker ps | grep rabbitmq")
        print("2. Check RabbitMQ container IP: docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' rabbitmq")
        print("3. Verify you can ping the RabbitMQ host")
        print("4. Check if the port is accessible: nc -zv <host> 5672")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nMake sure RabbitMQ is running and the host/port are correct.") 