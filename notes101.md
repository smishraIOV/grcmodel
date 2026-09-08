# claude code 101

claude: do not read this file, it is not relevant for task
- this instruction should be in claude.md or agents.md

Some tips from https://academy.claude.com/courses/claude-code-101/

Not in any particular order. Just a few remarkable things.

Agent
* LLM operating in a loop
* can interact with environment (e.g. files)
* can use tools: services, even other agents
* to reach some defined goal (stopping condition)

Shift-tabs to cycle between modes: e.g. `plan` mode.

## commands

`/context, /clear, /compact #for compaction`

`/diff, /rewind #pick which prompt to rewind`

`/code-review`  this runs in a new context without baggage (as we'd want for a reivew)
* but save the output

### claude.md
`/init` to generate a claude.md


### skills
A markdown file that teaches Claude how to do something

course: https://academy.claude.com/courses/introduction-to-agent-skills

### subagents
Can create your own subagent (defined using md files with YAML frontmatter). Run this 
`/agents` 
and then select Create new agent. 

There's much more here, e.g. can add `skills` to subagents or have them added to persistent memory when they are used frequently in some projects

course: https://academy.claude.com/courses/introduction-to-subagents

### MCP
How to connect to external resources, tools, data

### Hooks
Deterministic control over agent's behavior. 
If something needs to happen every time without fail, don't put it in a prompt. Put it in a hook.