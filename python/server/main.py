from flask import request, jsonify
from functools import wraps
from flask_sqlalchemy import SQLAlchemy

import flask
import os
import requests

app = flask.Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(" ")[1]
        else:
            token = None

        if token is None:
            return jsonify({'message': 'Token is missing'}), 403

        # Verify the token with Google
        try:
            # Google's token info endpoint
            r = requests.get(f'https://oauth2.googleapis.com/tokeninfo?id_token={token}')
            if r.status_code != 200:
                return jsonify({'message': 'Token is invalid'}), 403
            user_info = r.json()
            # You can now use user_info which contains the user's Google profile information
        except:
            return jsonify({'message': 'Token is invalid'}), 403

        return f(*args, **kwargs)

def get_session_id():
    return request.cookies.get('X-Session-Id')

@app.route('/api/projects', methods=['GET'])
@token_required
def fetchProjectsForUser():
    connection = None
    try:
        connection = db.engine.connect()
    except:
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    try:
        result = connection.execute('SELECT name, description, project_key FROM projects WHERE user_id = :session_id', {'session_id': session_id})
        projects = []
        for row in result:
            projects.append({
                'name': row['name'],
                'description': row['description'],
                'projectKey': row['project_key'],
            })
        return jsonify(projects)
    except:
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects', methods=['POST'])
@token_required
def upsertProject():
    connection = None
    try:
        connection = db.engine.connect()
    except:
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    try:
        body = request.get_json()
        if body['projectKey'] is None:
            # Insert
            connection.execute('INSERT INTO projects (name, description, user_id) VALUES (:name, :description, :session_id)', {'name': body['name'], 'description': body['description'], 'session_id': session_id})
        else:
            # Update
            connection.execute('UPDATE projects SET name = :name, description = :description WHERE project_key = :project_key AND user_id = :session_id', {'name': body['name'], 'description': body['description'], 'project_key': body['projectKey'], 'session_id': session_id})
        return jsonify()
    except:
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>', methods=['GET'])
@token_required
def fetchProject(project_key):
    connection = None
    try:
        connection = db.engine.connect()
    except:
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    try:
        result = connection.execute('SELECT name, description FROM projects WHERE project_key = :project_key AND user_id = :session_id LIMIT 1', {'project_key': project_key, 'session_id': session_id})
        if result.rowcount == 1:
            return jsonify({
                'name': row['name'],
                'description': row['description'],
                'projectKey': project_key,
            })
        elif result.rowcount == 0:
            return jsonify({'message': 'Project not found or not under this user'}), 404
        else:
            return jsonify({'message': f"More than one project with project_key={}"}), 500
    except:
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>', methods=['DELETE'])
@token_required
def deleteProject(project_key):
    connection = None
    try:
        connection = db.engine.connect()
    except:
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    try:
        result = connection.execute('DELETE FROM projects WHERE project_key = :project_key AND user_id = :session_id', {'project_key': project_key, 'session_id': session_id})
        if result.rowcount == 1:
            return jsonify()
        elif result.rowcount == 0:
            return jsonify({'message': 'Project not found or not under this user'}), 404
        else:
            return jsonify({'message': f"More than one project with project_key={}"}), 500
    except:
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>/datasets', methods=['GET'])
@token_required
def fetchDatasetsForProject(project_key):
    connection = None
    try:
        connection = db.engine.connect()
    except:
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    try:
        result = connection.execute('SELECT name, description, dataset_key FROM datasets WHERE project_key = :project_key AND user_id = :session_id', {'project_key': project_key, 'session_id': session_id})
        datasets = []
        for row in result:
            datasets.append({
                'name': row['name'],
                'description': row['description'],
                'datasetKey': row['dataset_key'],
                'projectKey': project_key,
            })
        return jsonify(datasets)
    except:
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>/datasets/<dataset_key>', methods=['GET'])
@token_required
def fetchDatasetData(project_key, dataset_key):
    connection = None
    try:
        connection = db.engine.connect()
    except:
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    try:
        result = connection.execute('SELECT name, description, data FROM datasets WHERE project_key = :project_key AND dataset_key = :dataset_key AND user_id = :session_id LIMIT 1', {'project_key': project_key, 'dataset_key': dataset_key, 'session_id': session_id})
        if result.rowcount == 1:
            return jsonify({
                'name': row['name'],
                'description': row['description'],
                'data': row['data'],
                'datasetKey': dataset_key,
                'projectKey': project_key,
            })
        elif result.rowcount == 0:
            return jsonify({'message': 'Dataset not found or not under this project'}), 404
        else:
            return jsonify({'message': f"More than one dataset with dataset_key={dataset_key} and project_key={project_key}"}), 500
    except:
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>/datasets', methods=['POST'])
@token_required
def upsertDataset(project_key):
    connection = None
    try:
        connection = db.engine.connect()
    except:
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    try:
        body = request.get_json()
        if body['datasetKey'] is None:
            # Insert
            connection.execute('INSERT INTO datasets (name, description, data, project_key, user_id) VALUES (:name, :description, :data, :project_key, :session_id)', {'name': body['name'], 'description': body['description'], 'data': body['data'], 'project_key': project_key, 'session_id': session_id})
        else:
            # Update
            connection.execute('UPDATE datasets SET name = :name, description = :description, data = :data WHERE dataset_key = :dataset_key AND project_key = :project_key AND user_id = :session_id', {'name': body['name'], 'description': body['description'], 'data': body['data'], 'dataset_key': body['datasetKey'], 'project_key': project_key, 'session_id': session_id})
        return jsonify()
    except:
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>/datasets/<dataset_key>', methods=['DELETE'])
@token_required
def deleteDataset(project_key, dataset_key):
    connection = None
    try:
        connection = db.engine.connect()
    except:
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    try:
        result = connection.execute('DELETE FROM datasets WHERE dataset_key = :dataset_key AND project_key = :project_key AND user_id = :session_id', {'dataset_key': dataset_key, 'project_key': project_key, 'session_id': session_id})
        if result.rowcount == 1:
            return jsonify()
        elif result.rowcount == 0:
            return jsonify({'message': 'Dataset not found or not under this project'}), 404
        else:
            return jsonify({'message': f"More than one dataset with dataset_key={dataset_key} and project_key={project_key}"}), 500
    except:
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>/datasets/<dataset_key>/append', methods=['POST'])
@token_required
def appendToDataset(project_key, dataset_key):
    connection = None
    try:
        connection = db.engine.connect()
    except:
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    try:
        body = request.get_json()
        result = connection.execute('SELECT data FROM datasets WHERE dataset_key = :dataset_key AND project_key = :project_key AND user_id = :session_id LIMIT 1', {'dataset_key': dataset_key, 'project_key': project_key, 'session_id': session_id})
        if result.rowcount == 1:
            row = result.fetchone()
            data = row['data']
            """
            in the format:
            {
                'data': [
                    ...
                ]
            }
            """
            data['data'].append(body['data'])
            connection.execute('UPDATE datasets SET data = :data WHERE dataset_key = :dataset_key AND project_key = :project_key AND user_id = :session_id', {'data': data, 'dataset_key': dataset_key, 'project_key': project_key, 'session_id': session_id})
            return jsonify()
        elif result.rowcount == 0:
            return jsonify({'message': 'Dataset not found or not under this project'}), 404
        else:
            return jsonify({'message': f"More than one dataset with dataset_key={dataset_key} and project_key={project_key}"}), 500
    except:
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>/lexicon', methods=['GET'])
@token_required
def fetchLexicons(project_key):
    connection = None
    try:
        connection = db.engine.connect()
    except:
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    try:
        result = connection.execute('SELECT project_key, src_language, tgt_language, data FROM lexicons WHERE project_key = :project_key AND user_id = :session_id', {'project_key': project_key, 'session_id': session_id})
        lexicons = []
        for row in result:
            lexicons.append({
                'projectKey': row['proejct_key'],
                'srcLanguage': row['src_language'],
                'tgtLanguage': row['tgt_language'],
                'data': row['data'],
            })
        return jsonify(lexicons)
    except:
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>/lexicon', methods=['POST'])
@token_required
def upsertLexicon(project_key):
    connection = None
    try:
        connection = db.engine.connect()
    except:
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    try:
        exists = connection.execute('SELECT count(*) FROM lexicons WHERE project_key = :project_key AND src_language = :src_language AND tgt_language = :tgt_language AND user_id = :session_id', {'project_key': project_key, 'src_language': body['srcLanguage'], 'tgt_language': body['tgtLanguage'], 'session_id': session_id})
        if exists.rowcount == 0:
            connection.execute('INSERT INTO lexicons (project_key, src_language, tgt_language, data, user_id) VALUES (:project_key, :src_language, :tgt_language, :data, :session_id)', {'project_key': project_key, 'src_language': body['srcLanguage'], 'tgt_language': body['tgtLanguage'], 'data': body['data'], 'session_id': session_id})
        else:
            # Update
            connection.execute('UPDATE lexicons SET src_language = :src_language, tgt_language = :tgt_language, data = :data WHERE project_key = :project_key AND user_id = :session_id', {'src_language': body['srcLanguage'], 'tgt_language': body['tgtLanguage'], 'data': body['data'], 'project_key': project_key, 'session_id': session_id})
        return jsonify()
    except:
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>/hyperparameters', methods=['GET'])
@token_required
def fetchHyperParameterConfigurations(project_key):
    connection = None
    try:
        connection = db.engine.connect()
    except:
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    try:
        result = connection.execute('''
        SELECT 
               name
              ,description
              ,project_key
              ,data
          FROM hyperparameter_configurations 
         WHERE project_key = :project_key 
           AND user_id = :session_id
        ''', {'project_key': project_key, 'session_id': session_id})
        configurations = []
        for row in result:
            data = row['data']
            configurations.append({
                'name': row['name'],
                'description': row['description'],
                'projectKey': row['project_key'],
                'hyperparametersKey': row['hyperparameters_key'],
                'tokenizer': data['tokenizer'],
                'src_vocab_size': data['src_vocab_size'],
                'tgt_vocab_size': data['tgt_vocab_size'],
                'shared_vocab': data['shared_vocab'],
                'max_length': data['max_length'],
                'min_length': data['min_length'],
                'max_length_ratio': data['max_length_ratio'],
                'd_model': data['d_model'],
                'n_heads': data['n_heads'],
                'd_queries': data['d_queries'],
                'd_values': data['d_values'],
                'd_inner': data['d_inner'],
                'n_encoder_layers': data['n_encoder_layers'],
                'n_decoder_layers': data['n_decoder_layers'],
                'dropout': data['dropout'],
                'positional_encoding_type': data['positional_encoding_type'],
                'rotary_positional_encoding_dim': data['rotary_positional_encoding_dim'],
                'tokens_in_batch': data['tokens_in_batch'],
                'target_tokens_per_batch': data['target_tokens_per_batch'],
                'n_steps': data['n_steps'],
                'warmup_steps': data['warmup_steps'],
                'beta1': data['beta1'],
                'beta2': data['beta2'],
                'epsilon': data['epsilon'],
                'label_smoothing': data['label_smoothing'],
            })
        return jsonify(configurations)

@app.route('/api/projects/<project_key>/hyperparameters', methods=['POST'])
@token_required
def upsertHyperparameterConfiguration(project_key):
    connection = None
    try:
        connection = db.engine.connect()
    except:
        return jsonify({'message': 'Database connection error'}), 500
    
    session_id = get_session_id()

    try:
        body = request.get_json()
        if body['hyperparametersKey'] is None:
            # Insert
            connection.execute('INSERT INTO hyperparameter_configurations (name, description, project_key, data, user_id) VALUES (:name, :description, :project_key, :data, :session_id)', {'name': body['name'], 'description': body['description'], 'project_key': project_key, 'data': body['data'], 'session_id': session_id})
        else:
            # Update
            connection.execute('UPDATE hyperparameter_configurations SET name = :name, description = :description, data = :data WHERE hyperparameters_key = :hyperparameters_key AND project_key = :project_key AND user_id = :session_id', {'name': body['name'], 'description': body['description'], 'data': body['data'], 'hyperparameters_key': body['hyperparametersKey'], 'project_key': project_key, 'session_id': session_id})
        return jsonify()
    except:
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>/hyperparameters/<hyperparameters_key>', methods=['DELETE'])

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
