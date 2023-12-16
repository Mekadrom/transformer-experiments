export default interface Project {
    name: string; // the name for this project (for display and logging purposes)
    description: string; // a description for this project, basically a place for users to store comments on the project
    projectKey: string | null; // a uuid reference to this project. null when new project
};
