// This file is the starting point for Jenkins continuous integration

// The continuous integration config is specified below (look at the file for more info)
// The config overlay starts with the first config, then updates field from consecutive configs
// Modify this if your branch needs a custom container, or custom yamlrun steps, etc.
import groovy.transform.Field
//@Field def config_overlay = ["ci/config/default.yaml"]
@Field def config_overlay = [
  "ci/config/default.yaml" : [],
  "ci/config/ocv3.yaml" : ["ci/config/default.yaml"],
]

// This is generic boilerplate used to bootstrap the jenkins scripts in the submodule
node("docker_plugin_node_2") {
  try {
    /**/// This checks out the code, and loads up the real pipeline driver from the submodule
    stage("Initialize Pipeline") {
      workspace_dir = pwd() + "/repo-ws/"
      repo_dir = workspace_dir + "/repo/"
      jenkins_dir = repo_dir + "/src/ci/jenkins/"
      dir(repo_dir + "/src/") {
        // checkout scm // < doesn't work with submodule credentials
	checkout([
	  $class: 'GitSCM',
          branches: scm.branches,
          doGenerateSubmoduleConfigurations: true,
          extensions: scm.extensions + [[$class: 'SubmoduleOption', parentCredentials: true, trackingSubmodules: true, recursiveSubmodules: true]],
          userRemoteConfigs: scm.userRemoteConfigs
        ])
      }
      main_ = fileLoader.load(jenkins_dir + "/jenkins/multi-pipeline-scripts/main")  
    }
    /*workspace_dir = pwd() + "/repo-ws/"
    repo_dir = workspace_dir + "/repo/"
    jenkins_dir = repo_dir + "/src/ci/jenkins/"
    main_ = fileLoader.load(jenkins_dir + "/jenkins/multi-pipeline-scripts/main")*/
    // This runs the main pipeline driver, we're done bootstrapping now
    main_.main_fn(config_overlay, repo_dir, jenkins_dir)
  } catch (any) {
    println "Something went wrong before calling the jenkins pipeline."
    throw any
  }
}

