from huggingface_hub import whoami, model_info
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError, LocalTokenNotFoundError

print("--- 1. Checking Login Status ---")
try:
    user_info = whoami()
    print(f"✅ Logged in as: {user_info['name']}")
    print(f"   Org memberships: {', '.join([org['name'] for org in user_info['orgs']]) if user_info['orgs'] else 'None'}")
except LocalTokenNotFoundError:
    print("❌ Not logged in. Please run `huggingface-cli login` first.")
    exit()
except Exception as e:
    print(f"❌ Login check failed: {e}")
    exit()

print("\n--- 2. Checking Access to SV3D ---")
repo_id = "stabilityai/sv3d-u"
try:
    # This attempts to fetch metadata. If you have access, it succeeds.
    info = model_info(repo_id)
    print(f"✅ SUCCESS! You have access to {repo_id}.")
    print(f"   Model Downloads: {info.downloads}")
except GatedRepoError:
    print(f"❌ FAILED: You are logged in, but have NOT accepted the license for {repo_id}.")
    print(f"   👉 Go here to accept it: https://huggingface.co/{repo_id}")
except RepositoryNotFoundError:
    print(f"❌ FAILED: The repo {repo_id} was not found.")
    print("   This usually means your token is invalid or missing 'Read' permissions.")
except Exception as e:
    print(f"❌ Unexpected error: {e}")