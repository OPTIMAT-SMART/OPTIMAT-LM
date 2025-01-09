from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/api/navigate', methods=['POST'])
def check_navigation():
    data = request.get_json()
    
    if not data or 'origin' not in data or 'destination' not in data:
        return jsonify({'error': 'Missing origin or destination'}), 400
        
    origin = data['origin'].lower()
    destination = data['destination'].lower()
    
    # Mock list of valid addresses - in a real app this would query a database
    valid_addresses = [
        "123 main st",
        "456 oak ave", 
        "789 pine rd",
        "321 elm st"
    ]
    
    if origin not in valid_addresses:
        return jsonify({'message': f'Invalid origin address: {origin}'}), 400
        
    if destination not in valid_addresses:
        return jsonify({'message': f'Invalid destination address: {destination}'}), 400
    
    return jsonify({'message': 'There is a path between origin and destination'})

if __name__ == '__main__':
    app.run(debug=True)
