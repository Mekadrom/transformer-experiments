export default interface Project {
    name: string; // the name for this project (for display and logging purposes)
    notes: string; // a place for notes
    projectKey: string | null; // a uuid reference to this project. null when new project
};
