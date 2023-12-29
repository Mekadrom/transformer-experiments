from flask import request, jsonify
from flask_cors import CORS
from functools import wraps
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

import flask
import json
import os
import psycopg2
import requests
import sys

POSTGRES_URL = os.getenv("POSTGRES_URL")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PW = os.getenv("POSTGRES_PW")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_SCHEMA = os.getenv("POSTGRES_SCHEMA")

DB_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PW}@{POSTGRES_URL}/{POSTGRES_DB}?options=-c%20search_path={POSTGRES_SCHEMA}"

app = flask.Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
db = SQLAlchemy(app)

train_device_dict = {}

def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(" ")[1]
        else:
            token = None

        if token is None:
            app.logger.error('Token is missing for request.')
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
            app.logger.error('Token is invalid for request.')
            return jsonify({'message': 'Token is invalid'}), 403

        return f(*args, **kwargs)
    return decorated_function

def get_session_id():
    if 'X-Session-Id' in request.headers:
        return str(request.headers.get('X-Session-Id'))
    else:
        return None

@app.route('/')
def index():
    return 'Hello world!', 200

@app.route('/api/projects', methods=['GET'])
@token_required
def fetch_all_projects_for_user():
    connection = None
    try:
        connection = db.engine.connect()
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    if session_id is None:
        app.logger.error('Session ID is missing for request.')
        return jsonify({'message': 'Session ID is missing. Please login.'}), 500

    try:
        result = connection.execute(text('SELECT project_key, name, description FROM myschema.projects WHERE user_id = :session_id'), {'session_id': session_id})
        projects = []
        for row in result:
            projects.append({
                'projectKey': row[0],
                'name': row[1],
                'description': row[2],
            })
        return jsonify(projects)
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects', methods=['POST'])
@token_required
def upsert_project():
    connection = None
    try:
        connection = db.engine.connect()
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    if session_id is None:
        app.logger.error('Session ID is missing for request.')
        return jsonify({'message': 'Session ID is missing. Please login.'}), 500

    try:
        body = request.get_json()
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Invalid request body'}), 400
    
    try:
        if 'projectKey' not in body or body['projectKey'] is None:
            # Insert
            app.logger.error('Inserting new project')
            connection.execute(text('INSERT INTO projects (name, description, user_id) VALUES (:name, :description, :session_id);').execution_options(autocommit=True), {'name': body['name'], 'description': body['description'], 'session_id': session_id})
        else:
            # Update
            app.logger.error('Updating project')
            connection.execute(text('UPDATE projects SET name = :name, description = :description WHERE project_key = :project_key AND user_id = :session_id').execution_options(autocommit=True), {'name': body['name'], 'description': body['description'], 'project_key': body['projectKey'], 'session_id': session_id})
        connection.commit()
        return jsonify({'message': 'Project created/updated successfully'}), 200
    except Exception as e:
        app.logger.error(e.with_traceback())
        connection.rollback()
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>', methods=['GET'])
@token_required
def fetch_project(project_key):
    connection = None
    try:
        connection = db.engine.connect()
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    if session_id is None:
        app.logger.error('Session ID is missing for request.')
        return jsonify({'message': 'Session ID is missing. Please login.'}), 500

    try:
        result = connection.execute(text('SELECT project_key, name, description FROM projects WHERE project_key = :project_key AND user_id = :session_id LIMIT 1'), {'project_key': project_key, 'session_id': session_id})
        if result.rowcount == 1:
            row = result.fetchone()
            return jsonify({
                'projectKey': row[0],
                'name': row[1],
                'description': row[2],
            })
        elif result.rowcount == 0:
            return jsonify({'message': 'Project not found or not under this user'}), 404
        else:
            return jsonify({'message': f"More than one project with project_key={project_key}"}), 500
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>', methods=['DELETE'])
@token_required
def delete_project(project_key):
    connection = None
    try:
        connection = db.engine.connect()
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    if session_id is None:
        app.logger.error('Session ID is missing for request.')
        return jsonify({'message': 'Session ID is missing. Please login.'}), 500

    try:
        result = connection.execute(text('DELETE FROM projects WHERE project_key = :project_key AND user_id = :session_id').execution_options(autocommit=True, autoflush=True), {'project_key': project_key, 'session_id': session_id})
        if result.rowcount == 1:
            return jsonify({'message': 'Project deleted'})
        elif result.rowcount == 0:
            return jsonify({'message': 'Project not found or not under this user'}), 404
        else:
            return jsonify({'message': f"More than one project with project_key={project_key}"}), 500
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>/lexicon', methods=['GET'])
@token_required
def fetch_all_lexicons_for_project(project_key):
    connection = None
    try:
        connection = db.engine.connect()
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    if session_id is None:
        app.logger.error('Session ID is missing for request.')
        return jsonify({'message': 'Session ID is missing. Please login.'}), 500

    try:
        result = connection.execute(text('SELECT project_key, src_language, tgt_language, data FROM lexicons WHERE project_key = :project_key AND user_id = :session_id'), {'project_key': project_key, 'session_id': session_id})
        lexicons = []
        for row in result:
            lexicons.append({
                'projectKey': row[0],
                'srcLanguage': row[1],
                'tgtLanguage': row[2],
                'data': json.loads(row[3]),
            })
        return jsonify(lexicons)
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>/lexicon', methods=['POST'])
@token_required
def upsert_lexicon(project_key):
    connection = None
    try:
        connection = db.engine.connect()
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    if session_id is None:
        app.logger.error('Session ID is missing for request.')
        return jsonify({'message': 'Session ID is missing. Please login.'}), 500

    try:
        body = request.get_json()
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Invalid request body'}), 400

    try:
        exists = connection.execute(text('SELECT count(*) FROM lexicons WHERE project_key = :project_key AND src_language = :src_language AND tgt_language = :tgt_language AND user_id = :session_id'), {'project_key': project_key, 'src_language': body['srcLanguage'], 'tgt_language': body['tgtLanguage'], 'session_id': session_id})
        if exists.rowcount == 0:
            connection.execute(text('INSERT INTO lexicons (project_key, src_language, tgt_language, data, user_id) VALUES (:project_key, :src_language, :tgt_language, :data, :session_id)').execution_options(autocommit=True), {'project_key': project_key, 'src_language': body['srcLanguage'], 'tgt_language': body['tgtLanguage'], 'data': json.dumps(body['data']), 'session_id': session_id})
        elif exists.rowcount == 1:
            # Update existing lexicon for project (only 1)
            connection.execute(text('UPDATE lexicons SET src_language = :src_language, tgt_language = :tgt_language, data = :data WHERE project_key = :project_key AND user_id = :session_id').execution_options(autocommit=True), {'src_language': body['srcLanguage'], 'tgt_language': body['tgtLanguage'], 'data': json.dumps(body['data']), 'project_key': project_key, 'session_id': session_id})
        else:
            return jsonify({'message': 'More than one lexicon found for this project'}), 500
        return jsonify({'message': 'Lexicon created/updated successfully'}), 200
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>/datasets', methods=['GET'])
@token_required
def fetch_all_datasets_for_project(project_key):
    connection = None
    try:
        connection = db.engine.connect()
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    if session_id is None:
        app.logger.error('Session ID is missing for request.')
        return jsonify({'message': 'Session ID is missing. Please login.'}), 500

    try:
        result = connection.execute(text('SELECT project_key, dataset_key, name, description FROM datasets WHERE project_key = :project_key AND user_id = :session_id'), {'project_key': project_key, 'session_id': session_id})
        datasets = []
        for row in result:
            datasets.append({
                'projectKey': row[0],
                'datasetKey': row[1],
                'name': row[2],
                'description': row[3],
            })
        return jsonify(datasets)
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>/datasets/<dataset_key>', methods=['GET'])
@token_required
def fetch_dataset_data(project_key, dataset_key):
    connection = None
    try:
        connection = db.engine.connect()
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    if session_id is None:
        app.logger.error('Session ID is missing for request.')
        return jsonify({'message': 'Session ID is missing. Please login.'}), 500

    try:
        result = connection.execute(text('SELECT project_key, dataset_key, name, description, data FROM datasets WHERE project_key = :project_key AND dataset_key = :dataset_key AND user_id = :session_id LIMIT 1'), {'project_key': project_key, 'dataset_key': dataset_key, 'session_id': session_id})
        if result.rowcount == 1:
            row = result.fetchone()
            return jsonify({
                'projectKey': row[0],
                'datasetKey': row[1],
                'name': row[2],
                'description': row[3],
                'data': row[4],
            })
        elif result.rowcount == 0:
            return jsonify({'message': 'Dataset not found or not under this project'}), 404
        else:
            return jsonify({'message': f"More than one dataset with dataset_key={dataset_key} and project_key={project_key}"}), 500
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>/datasets', methods=['POST'])
@token_required
def upsert_dataset(project_key):
    connection = None
    try:
        connection = db.engine.connect()
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    if session_id is None:
        app.logger.error('Session ID is missing for request.')
        return jsonify({'message': 'Session ID is missing. Please login.'}), 500

    try:
        body = request.get_json()
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Invalid request body'}), 400

    try:
        if 'datasetKey' not in body or body['datasetKey'] is None:
            # Insert
            connection.execute(text('INSERT INTO datasets (name, description, data, project_key, user_id) VALUES (:name, :description, :data, :project_key, :session_id)').execution_options(autocommit=True), {'name': body['name'], 'description': body['description'], 'data': body['data'], 'project_key': project_key, 'session_id': session_id})
        else:
            # Update
            connection.execute(text('UPDATE datasets SET name = :name, description = :description, data = :data WHERE dataset_key = :dataset_key AND project_key = :project_key AND user_id = :session_id').execution_options(autocommit=True), {'name': body['name'], 'description': body['description'], 'data': body['data'], 'dataset_key': body['datasetKey'], 'project_key': project_key, 'session_id': session_id})
        return jsonify()
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>/datasets/<dataset_key>', methods=['DELETE'])
@token_required
def delete_dataset(project_key, dataset_key):
    connection = None
    try:
        connection = db.engine.connect()
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    if session_id is None:
        app.logger.error('Session ID is missing for request.')
        return jsonify({'message': 'Session ID is missing. Please login.'}), 500

    try:
        result = connection.execute(text('DELETE FROM datasets WHERE dataset_key = :dataset_key AND project_key = :project_key AND user_id = :session_id').execution_options(autocommit=True), {'dataset_key': dataset_key, 'project_key': project_key, 'session_id': session_id})
        if result.rowcount == 1:
            return jsonify({'message': 'Dataset deleted'})
        elif result.rowcount == 0:
            return jsonify({'message': 'Dataset not found or not under this project'}), 404
        else:
            return jsonify({'message': f"More than one dataset with dataset_key={dataset_key} and project_key={project_key}"}), 500
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

@app.route('/api/projects/<project_key>/datasets/<dataset_key>/append', methods=['POST'])
@token_required
def append_to_dataset(project_key, dataset_key):
    connection = None
    try:
        connection = db.engine.connect()
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database connection error'}), 500

    session_id = get_session_id()

    if session_id is None:
        app.logger.error('Session ID is missing for request.')
        return jsonify({'message': 'Session ID is missing. Please login.'}), 500

    try:
        body = request.get_json()
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Invalid request body'}), 400

    try:
        result = connection.execute('SELECT data FROM datasets WHERE dataset_key = :dataset_key AND project_key = :project_key AND user_id = :session_id LIMIT 1', {'dataset_key': dataset_key, 'project_key': project_key, 'session_id': session_id})
        if result.rowcount == 1:
            row = result.fetchone()
            data = json.loads(row[0])
            """
            in the format:
            {
                'data': [
                    ...
                ]
            }
            """
            data['data'].append(body['data'])
            connection.execute(text('UPDATE datasets SET data = :data WHERE dataset_key = :dataset_key AND project_key = :project_key AND user_id = :session_id').execution_options(autocommit=True), {'data': data, 'dataset_key': dataset_key, 'project_key': project_key, 'session_id': session_id})
            return jsonify()
        elif result.rowcount == 0:
            return jsonify({'message': 'Dataset not found or not under this project'}), 404
        else:
            return jsonify({'message': f"More than one dataset with dataset_key={dataset_key} and project_key={project_key}"}), 500
    except Exception as e:
        app.logger.error(e)
        return jsonify({'message': 'Database query error'}), 500
    finally:
        connection.close()

# @app.route('/api/projects/<project_key>/hyperparameters', methods=['GET'])
# @token_required
# def fetch_all_hyperparameter_configurations_for_project(project_key):
#     connection = None
#     try:
#         connection = db.engine.connect()
#     except Exception as e:
#         app.logger.error(e)
#         return jsonify({'message': 'Database connection error'}), 500

#     session_id = get_session_id()

#     if session_id is None:
#         app.logger.error('Session ID is missing for request.')
#         return jsonify({'message': 'Session ID is missing. Please login.'}), 500

#     try:
#         result = connection.execute(text('''
#         SELECT 
#                project_key
#               ,hyperparameters_key
#               ,name
#               ,description
#               ,data
#           FROM hyperparameter_configurations 
#          WHERE project_key = :project_key 
#            AND user_id = :session_id
#         '''), {'project_key': project_key, 'session_id': session_id})
#         configurations = []
#         for row in result:
#             data = json.loads(row[4])
#             configurations.append({
#                 'projectKey': row[0],
#                 'hyperparametersKey': row[1],
#                 'name': row[2],
#                 'description': row[3],
#                 'tokenizer': data['tokenizer'],
#                 'src_vocab_size': data['src_vocab_size'],
#                 'tgt_vocab_size': data['tgt_vocab_size'],
#                 'shared_vocab': data['shared_vocab'],
#                 'max_length': data['max_length'],
#                 'min_length': data['min_length'],
#                 'max_length_ratio': data['max_length_ratio'],
#                 'd_model': data['d_model'],
#                 'n_heads': data['n_heads'],
#                 'd_queries': data['d_queries'],
#                 'd_values': data['d_values'],
#                 'd_inner': data['d_inner'],
#                 'n_encoder_layers': data['n_encoder_layers'],
#                 'n_decoder_layers': data['n_decoder_layers'],
#                 'dropout': data['dropout'],
#                 'positional_encoding_type': data['positional_encoding_type'],
#                 'rotary_positional_encoding_dim': data['rotary_positional_encoding_dim'],
#                 'tokens_in_batch': data['tokens_in_batch'],
#                 'target_tokens_per_batch': data['target_tokens_per_batch'],
#                 'n_steps': data['n_steps'],
#                 'warmup_steps': data['warmup_steps'],
#                 'beta1': data['beta1'],
#                 'beta2': data['beta2'],
#                 'epsilon': data['epsilon'],
#                 'label_smoothing': data['label_smoothing'],
#             })
#         if configurations.count == 0:
#             return jsonify({'message': 'No hyperparameter configurations found for this project'}), 404
#         return jsonify(configurations)
#     except Exception as e:
#         app.logger.error(e)
#         return jsonify({'message': 'Database query error'}), 500
#     finally:
#         connection.close()

# @app.route('/api/projects/<project_key>/hyperparameters/<hyperparameters_key>', methods=['GET'])
# @token_required
# def fetch_hyperparameter_configuration(project_key, hyperparameters_key):
#     connection = None
#     try:
#         connection = db.engine.connect()
#     except Exception as e:
#         app.logger.error(e)
#         return jsonify({'message': 'Database connection error'}), 500

#     session_id = get_session_id()

#     if session_id is None:
#         app.logger.error('Session ID is missing for request.')
#         return jsonify({'message': 'Session ID is missing. Please login.'}), 500
    
#     try:
#         result = connection.execute(text('''
#         SELECT 
#                project_key
#               ,hyperparameters_key
#               ,name
#               ,description
#               ,data
#           FROM hyperparameter_configurations 
#          WHERE project_key = :project_key 
#            AND hyperparameters_key = :hyperparameters_key
#            AND user_id = :session_id
#         '''), {'project_key': project_key, 'hyperparameters_key': hyperparameters_key, 'session_id': session_id})
#         if result.rowcount == 1:
#             row = result.fetchone()
#             data = json.loads(row[4])
#             return jsonify({
#                 'projectKey': row[0],
#                 'hyperparametersKey': row[1],
#                 'name': row[2],
#                 'description': row[3],
#                 'tokenizer': data['tokenizer'],
#                 'src_vocab_size': data['src_vocab_size'],
#                 'tgt_vocab_size': data['tgt_vocab_size'],
#                 'shared_vocab': data['shared_vocab'],
#                 'max_length': data['max_length'],
#                 'min_length': data['min_length'],
#                 'max_length_ratio': data['max_length_ratio'],
#                 'd_model': data['d_model'],
#                 'n_heads': data['n_heads'],
#                 'd_queries': data['d_queries'],
#                 'd_values': data['d_values'],
#                 'd_inner': data['d_inner'],
#                 'n_encoder_layers': data['n_encoder_layers'],
#                 'n_decoder_layers': data['n_decoder_layers'],
#                 'dropout': data['dropout'],
#                 'positional_encoding_type': data['positional_encoding_type'],
#                 'rotary_positional_encoding_dim': data['rotary_positional_encoding_dim'],
#                 'tokens_in_batch': data['tokens_in_batch'],
#                 'target_tokens_per_batch': data['target_tokens_per_batch'],
#                 'n_steps': data['n_steps'],
#                 'warmup_steps': data['warmup_steps'],
#                 'beta1': data['beta1'],
#                 'beta2': data['beta2'],
#                 'epsilon': data['epsilon'],
#                 'label_smoothing': data['label_smoothing'],
#             })
#         elif result.rowcount == 0:
#             return jsonify({'message': 'Hyperparameter configuration not found or not under this project'}), 404
#         else:
#             return jsonify({'message': f"More than one hyperparameter configuration with hyperparameters_key={hyperparameters_key} and project_key={project_key}"}), 500
#     except Exception as e:
#         app.logger.error(e)
#         return jsonify({'message': 'Database query error'}), 500
#     finally:
#         connection.close()

# @app.route('/api/projects/<project_key>/hyperparameters', methods=['POST'])
# @token_required
# def upsert_hyperparameter_configuration(project_key):
#     connection = None
#     try:
#         connection = db.engine.connect()
#     except Exception as e:
#         app.logger.error(e)
#         return jsonify({'message': 'Database connection error'}), 500
    
#     session_id = get_session_id()

#     if session_id is None:
#         app.logger.error('Session ID is missing for request.')
#         return jsonify({'message': 'Session ID is missing. Please login.'}), 500

#     try:
#         body = request.get_json()
#     except Exception as e:
#         app.logger.error(e)
#         return jsonify({'message': 'Invalid request body'}), 400

#     try:
#         if 'hyperparametersKey' not in body or body['hyperparametersKey'] is None:
#             # Insert
#             connection.execute(text('INSERT INTO hyperparameter_configurations (name, description, project_key, data, user_id) VALUES (:name, :description, :project_key, :data, :session_id)').execution_options(autocommit=True), {'name': body['name'], 'description': body['description'], 'project_key': project_key, 'data': body['data'], 'session_id': session_id})
#         else:
#             # Update
#             connection.execute(text('UPDATE hyperparameter_configurations SET name = :name, description = :description, data = :data WHERE hyperparameters_key = :hyperparameters_key AND project_key = :project_key AND user_id = :session_id').execution_options(autcommit=True), {'name': body['name'], 'description': body['description'], 'data': body['data'], 'hyperparameters_key': body['hyperparametersKey'], 'project_key': project_key, 'session_id': session_id})
#         return jsonify()
#     except Exception as e:
#         app.logger.error(e)
#         return jsonify({'message': 'Database query error'}), 500
#     finally:
#         connection.close()

# @app.route('/api/projects/<project_key>/hyperparameters/<hyperparameters_key>', methods=['DELETE'])
# @token_required
# def delete_hyperparameter_configuration(hyperparameters_key):
#     connection = None
#     try:
#         connection = db.engine.connect()
#     except Exception as e:
#         app.logger.error(e)
#         return jsonify({'message': 'Database connection error'}), 500
    
#     session_id = get_session_id()

#     if session_id is None:
#         app.logger.error('Session ID is missing for request.')
#         return jsonify({'message': 'Session ID is missing. Please login.'}), 500

#     try:
#         result = connection.execute(text('DELETE FROM hyperparameter_configurations WHERE hyperparameters_key = :hyperparameters_key AND user_id = :session_id').execution_options(autocommit=True), {'hyperparameters_key': hyperparameters_key, 'session_id': session_id})
#         if result.rowcount == 1:
#             return jsonify({'message': 'Hyperparameter configuration deleted'})
#         elif result.rowcount == 0:
#             return jsonify({'message': 'Hyperparameter configuration not found or not under this project'}), 404
#         else:
#             return jsonify({'message': f"More than one hyperparameter configuration with hyperparameters_key={hyperparameters_key}"}), 500
#     except Exception as e:
#         app.logger.error(e)
#         return jsonify({'message': 'Database query error'}), 500
#     finally:
#         connection.close()

# @app.route('/api/projects/<project_key>/trainedModels', methods=['GET'])
# @token_required
# def get_all_trained_models_for_project(project_key):
#     connection = None
#     try:
#         connection = db.engine.connect()
#     except Exception as e:
#         app.logger.error(e)
#         return jsonify({'message': 'Database connection error'}), 500
    
#     session_id = get_session_id()

#     if session_id is None:
#         app.logger.error('Session ID is missing for request.')
#         return jsonify({'message': 'Session ID is missing. Please login.'}), 500

#     try:
#         result = connection.execute(text('''
#         SELECT 
#                project_key
#               ,hyperparameters_key
#               ,model_key
#               ,name
#               ,description
#               ,data
#           FROM trained_models 
#          WHERE project_key = :project_key 
#            AND user_id = :session_id
#         '''), {'project_key': project_key, 'session_id': session_id})
#         models = []
#         for row in result:
#             models.append({
#                 'projectKey': row[0],
#                 'hyperparametersKey': row[1],
#                 'modelKey': row[2],
#                 'name': row[3],
#                 'description': row[4],
#                 'data': json.loads(row[5]),
#             })

#         if models.count == 0:
#             return jsonify({'message': 'No trained models found for this project'}), 404
#         return jsonify(models)
#     except Exception as e:
#         app.logger.error(e)
#         return jsonify({'message': 'Database query error'}), 500
#     finally:
#         connection.close()

# @app.route('/api/projects/<project_key>/trainedModels/<model_key>', methods=['GET'])
# @token_required
# def get_trained_model(project_key, model_key):
#     connection = None
#     try:
#         connection = db.engine.connect()
#     except Exception as e:
#         app.logger.error(e)
#         return jsonify({'message': 'Database connection error'}), 500
    
#     session_id = get_session_id()

#     if session_id is None:
#         app.logger.error('Session ID is missing for request.')
#         return jsonify({'message': 'Session ID is missing. Please login.'}), 500

#     try:
#         result = connection.execute(text('''
#         SELECT 
#                project_key
#               ,model_key
#               ,hyperparameters_key
#               ,name
#               ,description
#           FROM trained_models 
#          WHERE project_key = :project_key 
#            AND model_key = :model_key
#            AND user_id = :session_id
#         '''), {'project_key': project_key, 'model_key': model_key, 'session_id': session_id})
#         if result.rowcount == 1:
#             row = result.fetchone()
#             data = json.loads(row[5])
#             return jsonify({
#                 'projectKey': row[0],
#                 'hyperparametersKey': row[1],
#                 'modelKey': row[2],
#                 'name': row[3],
#                 'description': row[4],
#             })
#         elif result.rowcount == 0:
#             return jsonify({'message': 'Model not found or not under this project'}), 404
#         else:
#             return jsonify({'message': f"More than one model with model_key={model_key} and project_key={project_key}"}), 500
#     except Exception as e:
#         app.logger.error(e)
#         return jsonify({'message': 'Database query error'}), 500
#     finally:
#         connection.close()

# @app.route('/api/projects/<project_key>/trainedModels/<model_key>', methods=['DELETE'])
# @token_required
# def delete_trained_model(project_key, model_key):
#     connection = None
#     try:
#         connection = db.engine.connect()
#     except Exception as e:
#         app.logger.error(e)
#         return jsonify({'message': 'Database connection error'}), 500
    
#     session_id = get_session_id()

#     if session_id is None:
#         app.logger.error('Session ID is missing for request.')
#         return jsonify({'message': 'Session ID is missing. Please login.'}), 500

#     try:
#         result = connection.execute(text('DELETE FROM trained_models WHERE model_key = :model_key AND project_key = :project_key AND user_id = :session_id').execution_options(autocommit=True), {'model_key': model_key, 'project_key': project_key, 'session_id': session_id})
#         if result.rowcount == 1:
#             return jsonify({'message': 'Trained model deleted'})
#         elif result.rowcount == 0:
#             return jsonify({'message': 'Trained model not found or not under this project'}), 404
#         else:
#             return jsonify({'message': f"More than one trained model with model_key={model_key} and project_key={project_key}"}), 500
#     except Exception as e:
#         app.logger.error(e)
#         return jsonify({'message': 'Database query error'}), 500
#     finally:
#         connection.close()

# @app.route('/api/projects/<project_key>/trainedModels/<model_key>/predict', methods=['POST'])
# @token_required
# def predict(project_key, model_key):
#     pass

if __name__ == '__main__':
    app.run(host=os.getenv('SERVER_HOST_URL') or '0.0.0.0', port=os.getenv('SERVER_PORT') or 5000, debug=True)
